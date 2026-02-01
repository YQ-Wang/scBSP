"""
@author: lijinp yiqingwang

This module utilizes a granularity-based dimension-agnostic tool, single-cell
big-small patch (scBSP), implementing sparse matrix operation for distance
calculation, for the identification of spatially variable genes on
large-scale data.
"""

from typing import List, Union, Optional

import numpy as np
import pandas as pd  # type: ignore
import scipy  # type: ignore
from scipy.sparse import csr_matrix, diags, identity, isspmatrix_csr  # type: ignore
from scipy.stats import gmean, lognorm, chi2, norm  # type: ignore
from sklearn.neighbors import BallTree  # type: ignore

gpu_enabled = True
gpu_backend = "torch_sparse"

try:
    import torch  # type: ignore
    if not torch.cuda.is_available():
        gpu_enabled = False
        gpu_backend = None
        print("CUDA is not available, using CPU instead.")
except ImportError:
    gpu_enabled = False
    gpu_backend = None


def _scale_sparse_matrix(input_exp_mat: csr_matrix) -> csr_matrix:
    """
    Scales a sparse matrix such that each row is divided by its maximum value.

    Args:
        input_exp_mat: A csr_matrix representing the input expression matrix.

    Returns:
        A csr_matrix scaled by row maximums.
    """

    if input_exp_mat.shape[0] == 0 or input_exp_mat.shape[1] == 0:
        return input_exp_mat

    data = input_exp_mat.data.copy()
    rows, cols = input_exp_mat.nonzero()

    row_indices = np.diff(input_exp_mat.indptr)
    row_idx = np.r_[0, np.cumsum(row_indices)]

    # Vectorized row max computation
    non_empty_mask = row_indices > 0
    row_max = np.ones(input_exp_mat.shape[0], dtype=data.dtype)

    if non_empty_mask.any():
        non_empty_starts = row_idx[:-1][non_empty_mask]
        row_max[non_empty_mask] = np.maximum.reduceat(data, non_empty_starts)

    data_scaled = data / np.repeat(row_max, row_indices)
    scaled_matrix = csr_matrix((data_scaled, (rows, cols)), shape=input_exp_mat.shape)

    return scaled_matrix


def _binary_distance_matrix_threshold(
    input_sparse_mat_array: np.ndarray, d_val: float, leaf_size: int
) -> csr_matrix:
    """
    Creates a binary distance matrix where distances below a threshold are marked as 1.

    Args:
        input_sparse_mat_array: The input sparse matrix array.
        d_val: The distance threshold.
        leaf_size: An integer for BallTree.

    Returns:
        A csr_matrix representing the binary distance matrix.
    """
    ball_tree = BallTree(input_sparse_mat_array, leaf_size=leaf_size)
    indices = ball_tree.query_radius(
        input_sparse_mat_array, r=d_val, return_distance=False
    )
    
    lengths = np.array([len(idx) for idx in indices])
    total_nnz = lengths.sum()
    rows = np.repeat(np.arange(len(indices)), lengths)
    cols = np.concatenate(indices) if total_nnz > 0 else np.array([], dtype=np.intp)
    data = np.ones(total_nnz, dtype=np.int8)

    return csr_matrix(
        (data, (rows, cols)),
        shape=(input_sparse_mat_array.shape[0], input_sparse_mat_array.shape[0]),
        dtype=np.int8
    )


def _calculate_sparse_variances(input_csr_mat: csr_matrix, axis: int) -> np.matrix:
    """Calculates the variances along a given axis for a csr_matrix."""
    input_csr_mat_squared = input_csr_mat.copy()
    input_csr_mat_squared.data **= 2
    return input_csr_mat_squared.mean(axis) - np.square(input_csr_mat.mean(axis))


def _get_test_scores(
    input_sp_mat: np.ndarray,
    input_exp_mat_raw: csr_matrix,
    d1: float,
    d2: float,
    leaf_size: int,
    use_gpu: bool,
) -> List[float]:
    """Calculates test scores for genomic data."""
    input_exp_mat_norm = _scale_sparse_matrix(input_exp_mat_raw).transpose()
    input_exp_mat_raw = input_exp_mat_raw.transpose()

    def _var_local_means(
        input_sp_mat: np.ndarray,
        d_val: float,
        input_exp_mat_norm: csr_matrix,
        leaf_size: int,
        use_gpu: bool
    ) -> np.matrix:
        patches_cells = _binary_distance_matrix_threshold(
            input_sp_mat, d_val, leaf_size
        )
        
        # Scaling patches
        sum_axis_0 = patches_cells.sum(axis=0).A.ravel().astype(np.float64)
        inv_sum = np.reciprocal(sum_axis_0, where=sum_axis_0 != 0, out=np.zeros_like(sum_axis_0, dtype=np.float64))
        patches_scaled = patches_cells.astype(np.float64).multiply(inv_sum)

        if use_gpu and gpu_enabled:
            import warnings
            warnings.filterwarnings('ignore', category=UserWarning)
            
            try:
                # Memory-efficient GPU multiplication strategy: (Patches^T @ Exp^T)^T
                # This avoids O(Cells^2) dense matrix transfers by keeping patches sparse.
                exp_t_dense_gpu = torch.tensor(input_exp_mat_norm.toarray().T.astype(np.float32), device='cuda')
                
                patches_t_sparse_gpu = torch.sparse_csr_tensor(
                    torch.tensor(patches_cells.T.indptr, dtype=torch.int64, device='cuda'),
                    torch.tensor(patches_cells.T.indices, dtype=torch.int64, device='cuda'),
                    torch.tensor(patches_cells.T.data.astype(np.float32), device='cuda'),
                    size=patches_cells.T.shape
                )
                
                # Sparse-Dense Multiplication on GPU
                res_t_gpu = torch.sparse.mm(patches_t_sparse_gpu, exp_t_dense_gpu)
                
                # Scale and calculate statistics
                inv_sum_gpu = torch.tensor(inv_sum.astype(np.float32), device='cuda').view(-1, 1)
                res_t_gpu *= inv_sum_gpu
                
                mean_x = res_t_gpu.mean(dim=0)
                mean_x2 = (res_t_gpu**2).mean(dim=0)
                var_gpu = mean_x2 - mean_x**2
                
                result_vars = var_gpu.cpu().numpy()
                
                del res_t_gpu, exp_t_dense_gpu, patches_t_sparse_gpu, inv_sum_gpu, var_gpu, mean_x, mean_x2
                
                return np.matrix(result_vars).T
                
            except Exception as e:
                print(f"GPU optimization failed, falling back to CPU: {e}")
                x_kj = input_exp_mat_norm @ patches_scaled
        else:
            x_kj = input_exp_mat_norm @ patches_scaled

        return _calculate_sparse_variances(x_kj, axis=1)

    def var_x_generator():
        for d_val in (d1, d2):
            yield _var_local_means(input_sp_mat, d_val, input_exp_mat_norm, leaf_size, use_gpu).A.ravel()

    var_x = np.column_stack(list(var_x_generator()))
    var_x_0_add = _calculate_sparse_variances(input_exp_mat_raw, axis=1).A.ravel()
    var_x_0_add /= max(var_x_0_add)
    t_matrix = (var_x[:, 1] / var_x[:, 0]) * var_x_0_add
    return t_matrix.tolist()


def granp(
    input_sp_mat: np.ndarray,
    input_exp_mat_raw: Union[np.ndarray, pd.DataFrame, csr_matrix],
    d1: float = 1.0,
    d2: float = 3.0,
    leaf_size: int = 80,
    use_gpu: bool = False
) -> pd.DataFrame:
    """Calculates the p-values for genomic data."""
    if isinstance(input_exp_mat_raw, pd.DataFrame):
        gene_names = input_exp_mat_raw.columns.astype(str).tolist()
        input_exp_mat_raw = csr_matrix(input_exp_mat_raw)
    else:
        gene_names = [f"Gene_{i}" for i in range(input_exp_mat_raw.shape[1])]
        input_exp_mat_raw = (
            input_exp_mat_raw
            if isspmatrix_csr(input_exp_mat_raw)
            else csr_matrix(input_exp_mat_raw)
        )

    scale_factor = (
        gmean(
            np.quantile(input_sp_mat, 0.975, axis=0)
            - np.quantile(input_sp_mat, 0.025, axis=0)
        )
        / 0.95
        / (input_sp_mat.shape[0]) ** (1 / input_sp_mat.shape[1])
    )
    d1 *= scale_factor
    d2 *= scale_factor

    t_matrix_sum = _get_test_scores(input_sp_mat, input_exp_mat_raw, d1, d2, leaf_size, use_gpu)

    t_matrix_array = np.asarray(t_matrix_sum)
    t_matrix_sum_upper90 = np.quantile(t_matrix_array, 0.90)
    mask = t_matrix_array < t_matrix_sum_upper90
    log_t_matrix_sum_mid = np.log(t_matrix_array[mask])
    log_norm_params = (log_t_matrix_sum_mid.mean(), log_t_matrix_sum_mid.std(ddof=1))

    p_values = 1 - lognorm.cdf(t_matrix_array, scale=np.exp(log_norm_params[0]), s=log_norm_params[1])
    return pd.DataFrame({"gene_names": gene_names, "p_values": p_values.tolist()})


def combine_p_values(
    list_of_pvalues: List[pd.DataFrame],
    method: str = "fisher"
) -> pd.DataFrame:
    """Combines p-values across multiple samples using Fisher's or Stouffer's method."""
    if method not in ["fisher", "stouffer"]:
        raise ValueError(f"Method must be 'fisher' or 'stouffer', got '{method}'")
    
    if not list_of_pvalues:
        return pd.DataFrame(columns=['gene_names', 'number_samples', 'calibrated_p_values'])

    # Validate input DataFrames
    for i, df in enumerate(list_of_pvalues):
        if not isinstance(df, pd.DataFrame):
            raise ValueError(f"Element {i} in list_of_pvalues is not a DataFrame")
        if 'gene_names' not in df.columns or 'p_values' not in df.columns:
            raise ValueError(f"DataFrame {i} must have 'gene_names' and 'p_values' columns")
    
    dfs_renamed = []
    for i, df in enumerate(list_of_pvalues):
        df_copy = df.copy()
        df_copy = df_copy.rename(columns={'p_values': f'p_values_{i+1}'})
        dfs_renamed.append(df_copy)
    
    merged = dfs_renamed[0]
    for df in dfs_renamed[1:]:
        merged = pd.merge(merged, df, on='gene_names', how='outer')
    
    pval_cols = [col for col in merged.columns if col.startswith('p_values_')]
    gene_names = merged['gene_names'].to_numpy()
    pval_matrix = merged[pval_cols].to_numpy()
    valid_counts = np.sum(~np.isnan(pval_matrix), axis=1)
    combined_pvals = np.full(len(gene_names), np.nan)
    valid_mask = valid_counts > 0

    if valid_mask.any():
        if method == "fisher":
            epsilon = 1e-300
            pval_safe = np.where(np.isnan(pval_matrix), 1.0, np.maximum(pval_matrix, epsilon))
            stat = -2 * np.sum(np.log(pval_safe), axis=1)
            combined_pvals[valid_mask] = 1 - chi2.cdf(stat[valid_mask], 2 * valid_counts[valid_mask])
        elif method == "stouffer":
            pval_clipped = np.clip(pval_matrix, 1e-15, 1 - 1e-15)
            z_scores = norm.ppf(1 - pval_clipped / 2) * np.sign(0.5 - pval_clipped)
            z_scores = np.where(np.isnan(pval_matrix), 0, z_scores)
            z_combined = np.sum(z_scores, axis=1) / np.sqrt(np.maximum(valid_counts, 1))
            combined_pvals[valid_mask] = 2 * (1 - norm.cdf(np.abs(z_combined[valid_mask])))

    return pd.DataFrame({
        'gene_names': gene_names,
        'number_samples': valid_counts.astype(int),
        'calibrated_p_values': combined_pvals
    })
