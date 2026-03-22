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
from scipy.sparse import csr_matrix, isspmatrix_csr  # type: ignore
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

    row_lengths = np.diff(input_exp_mat.indptr)
    nnz_starts = input_exp_mat.indptr[:-1]

    non_empty_mask = row_lengths > 0
    row_max = np.ones(input_exp_mat.shape[0], dtype=data.dtype)

    if non_empty_mask.any():
        row_max[non_empty_mask] = np.maximum.reduceat(data, nnz_starts[non_empty_mask])

    data_scaled = data / np.repeat(row_max, row_lengths)
    scaled_matrix = csr_matrix(
        (data_scaled, input_exp_mat.indices.copy(), input_exp_mat.indptr.copy()),
        shape=input_exp_mat.shape,
    )

    return scaled_matrix


def _binary_distance_matrix_threshold(
    input_sparse_mat_array: np.ndarray, d_val: float, leaf_size: int,
    ball_tree: Optional[BallTree] = None
) -> csr_matrix:
    """
    Creates a binary distance matrix where distances below a threshold are marked as 1.

    Args:
        input_sparse_mat_array: The input sparse matrix array.
        d_val: The distance threshold.
        leaf_size: An integer for BallTree.
        ball_tree: Pre-built BallTree to reuse. Built from input_sparse_mat_array if None.

    Returns:
        A csr_matrix representing the binary distance matrix.
    """
    if ball_tree is None:
        ball_tree = BallTree(input_sparse_mat_array, leaf_size=leaf_size)
    indices = ball_tree.query_radius(
        input_sparse_mat_array, r=d_val, return_distance=False
    )

    lengths = np.array([len(idx) for idx in indices])
    total_nnz = lengths.sum()
    indptr = np.empty(len(indices) + 1, dtype=np.int64)
    indptr[0] = 0
    np.cumsum(lengths, out=indptr[1:])
    col_indices = np.concatenate(indices) if total_nnz > 0 else np.array([], dtype=np.intp)
    data = np.ones(total_nnz, dtype=np.int8)
    n = input_sparse_mat_array.shape[0]

    mat = csr_matrix((data, col_indices, indptr), shape=(n, n), dtype=np.int8)
    mat.sort_indices()
    return mat


def _calculate_sparse_variances(input_csr_mat: csr_matrix, axis: int) -> np.ndarray:
    """Calculates the variances along a given axis for a csr_matrix."""
    input_csr_mat_squared = input_csr_mat.copy()
    input_csr_mat_squared.data **= 2
    result = input_csr_mat_squared.mean(axis) - np.square(input_csr_mat.mean(axis))
    return np.asarray(result).ravel() if axis == 0 else np.asarray(result)


def _get_test_scores(
    input_sp_mat: np.ndarray,
    input_exp_mat_raw: csr_matrix,
    d1: float,
    d2: float,
    leaf_size: int,
    use_gpu: bool,
) -> np.ndarray:
    """Calculates test scores for genomic data."""
    input_exp_mat_norm = _scale_sparse_matrix(input_exp_mat_raw).T
    input_exp_mat_raw = input_exp_mat_raw.T

    ball_tree = BallTree(input_sp_mat, leaf_size=leaf_size)

    def _var_local_means(
        input_sp_mat: np.ndarray,
        d_val: float,
        input_exp_mat_norm: csr_matrix,
        ball_tree: BallTree,
        use_gpu: bool
    ) -> np.ndarray:
        patches_cells = _binary_distance_matrix_threshold(
            input_sp_mat, d_val, leaf_size, ball_tree=ball_tree
        )

        if use_gpu and gpu_enabled:
            import warnings
            with warnings.catch_warnings():
                warnings.filterwarnings('ignore', category=UserWarning)

                try:
                    col_counts = np.asarray(patches_cells.getnnz(axis=0), dtype=np.float32)
                    inv_sum = np.reciprocal(col_counts, where=col_counts != 0, out=np.zeros_like(col_counts))

                    patches_t_csr = patches_cells.T.tocsr()
                    N_genes = input_exp_mat_norm.shape[0]
                    M_cells = input_exp_mat_norm.shape[1]
                    batch_size = max(1, 250_000_000 // M_cells)

                    patches_t_sparse_gpu = torch.sparse_csr_tensor(
                        torch.from_numpy(patches_t_csr.indptr.astype(np.int64)).to('cuda'),
                        torch.from_numpy(patches_t_csr.indices.astype(np.int64)).to('cuda'),
                        torch.from_numpy(patches_t_csr.data.astype(np.float32)).to('cuda'),
                        size=patches_t_csr.shape
                    )
                    del patches_t_csr

                    inv_sum_gpu = torch.from_numpy(inv_sum).to('cuda').view(-1, 1)
                    
                    result_vars_list = []
                    for start_idx in range(0, N_genes, batch_size):
                        end_idx = min(start_idx + batch_size, N_genes)
                        
                        exp_batch_np = input_exp_mat_norm[start_idx:end_idx, :].toarray().T.astype(np.float32)
                        exp_batch_gpu = torch.from_numpy(exp_batch_np).to('cuda')
                        
                        res_batch_gpu = torch.sparse.mm(patches_t_sparse_gpu, exp_batch_gpu)
                        del exp_batch_gpu
                        
                        res_batch_gpu *= inv_sum_gpu
                        
                        mean_x = res_batch_gpu.mean(dim=0)
                        mean_x2 = res_batch_gpu.square_().mean(dim=0)
                        del res_batch_gpu
                        
                        var_batch_gpu = mean_x2 - mean_x ** 2
                        del mean_x, mean_x2
                        
                        result_vars_list.append(var_batch_gpu.cpu().numpy())
                        del var_batch_gpu

                    del inv_sum_gpu, patches_t_sparse_gpu
                    torch.cuda.empty_cache()

                    result_vars = np.concatenate(result_vars_list)
                    return result_vars.reshape(-1, 1)

                except Exception as e:
                    print(f"GPU optimization failed, falling back to CPU: {e}")

        col_counts = np.asarray(patches_cells.getnnz(axis=0), dtype=np.float64)
        inv_sum = np.reciprocal(col_counts, where=col_counts != 0, out=np.zeros_like(col_counts))
        patches_scaled = patches_cells.astype(np.float64).multiply(inv_sum)
        
        N_genes = input_exp_mat_norm.shape[0]
        M_cells = input_exp_mat_norm.shape[1]
        batch_size = max(1, 10_000_000 // M_cells)
        
        result_vars_list = []
        for start_idx in range(0, N_genes, batch_size):
            end_idx = min(start_idx + batch_size, N_genes)
            exp_batch = input_exp_mat_norm[start_idx:end_idx, :]
            
            x_kj_batch = exp_batch @ patches_scaled
            var_batch = _calculate_sparse_variances(x_kj_batch, axis=1)
            result_vars_list.append(var_batch)

        return np.concatenate(result_vars_list).reshape(-1, 1) if result_vars_list else np.array([]).reshape(-1, 1)

    def var_x_generator():
        for d_val in (d1, d2):
            result = _var_local_means(input_sp_mat, d_val, input_exp_mat_norm, ball_tree, use_gpu)
            yield result.ravel()

    var_x = np.column_stack(list(var_x_generator()))
    var_x_0_add = _calculate_sparse_variances(input_exp_mat_raw, axis=1).ravel()
    var_x_0_add /= max(var_x_0_add)
    t_matrix = np.divide(
        var_x[:, 1], var_x[:, 0],
        out=np.zeros_like(var_x[:, 1]),
        where=var_x[:, 0] != 0
    ) * var_x_0_add
    return t_matrix


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

    t_matrix = _get_test_scores(input_sp_mat, input_exp_mat_raw, d1, d2, leaf_size, use_gpu)

    t_matrix_upper90 = np.quantile(t_matrix, 0.90)
    mask = t_matrix < t_matrix_upper90
    log_t_mid = np.log(t_matrix[mask])
    log_norm_params = (log_t_mid.mean(), log_t_mid.std(ddof=1))

    p_values = 1 - lognorm.cdf(t_matrix, scale=np.exp(log_norm_params[0]), s=log_norm_params[1])
    return pd.DataFrame({"gene_names": gene_names, "p_values": p_values})


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
    
    dfs_indexed = []
    for i, df in enumerate(list_of_pvalues):
        df_indexed = df.set_index('gene_names').rename(columns={'p_values': f'p_values_{i+1}'})
        dfs_indexed.append(df_indexed)
    
    merged = pd.concat(dfs_indexed, axis=1, join='outer').reset_index(names='gene_names')
    
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
