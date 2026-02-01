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
gpu_backend = None  # "torch_sparse", "torch_dense", or "cupy"
try:
    import torch  # type: ignore
    if torch.cuda.is_available():
        gpu_backend = "torch_sparse"
    else:
        gpu_enabled = False
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

    data = input_exp_mat.data
    rows, cols = input_exp_mat.nonzero()

    row_indices = np.diff(input_exp_mat.indptr)
    row_idx = np.r_[0, np.cumsum(row_indices)]

    row_max = np.array(
        [
            data[start:end].max() if end > start else 1
            for start, end in zip(row_idx[:-1], row_idx[1:])
        ]
    )

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
        leaf_size: An integer that determines the maximum number of points after which the Ball Tree algorithm opts for a brute-force search approach.

    Returns:
        A csr_matrix representing the binary distance matrix.
    """

    ball_tree = BallTree(input_sparse_mat_array, leaf_size=leaf_size)
    indices = ball_tree.query_radius(
        input_sparse_mat_array, r=d_val, return_distance=False
    )
    
    def generate_data():
        for i, idx in enumerate(indices):
            yield from ((i, j, 1) for j in idx)

    rows, cols, data = zip(*generate_data())
    
    sparse_mat = csr_matrix(
        (data, (rows, cols)),
        shape=(input_sparse_mat_array.shape[0], input_sparse_mat_array.shape[0]),
        dtype=np.int8
    )

    return sparse_mat + identity(
        input_sparse_mat_array.shape[0], format="csr", dtype=np.int8
    )


def _calculate_sparse_variances(input_csr_mat: csr_matrix, axis: int) -> np.matrix:
    """
    Calculates the variances along a given axis for a csr_matrix.

    Args:
        input_csr_mat: The input CSR matrix.
        axis: The axis along which the variances are calculated.

    Returns:
        A list of variances for each dimension along the specified axis.
    """

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
    """
    Calculates test scores for genomic data based on input sparse matrices and distance thresholds.

    Args:
        input_sp_mat: The input spatial matrix as a numpy array.
        input_exp_mat_raw: The raw expression matrix in csr_matrix format.
        d1: Distance threshold 1.
        d2: Distance threshold 2.
        leaf_size: An integer that determines the maximum number of points after which the Ball Tree algorithm opts for a brute-force search approach.
        use_gpu: A boolean value that determines whether to use the GPU.

    Returns:
        A list of test scores.
    """

    input_exp_mat_norm = _scale_sparse_matrix(input_exp_mat_raw).transpose()
    input_exp_mat_raw = input_exp_mat_raw.transpose()

    def _get_inverted_diag_matrix(sum_axis_0: np.ndarray) -> csr_matrix:
        diag_data = np.zeros_like(sum_axis_0, dtype=np.float32)
        with np.errstate(divide="ignore", invalid="ignore"):
            np.reciprocal(sum_axis_0, where=sum_axis_0 != 0, out=diag_data)
        return diags(diag_data, offsets=0, format="csr")

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
        patches_cells_centroid = diags(
            (patches_cells.sum(axis=1) > 1).astype(np.float32).A.ravel(),
            offsets=0,
            format="csr",
        )
        patches_cells -= patches_cells_centroid
        sum_axis_0 = patches_cells.sum(axis=0).A.ravel()
        diag_matrix_sparse = _get_inverted_diag_matrix(sum_axis_0)

        if not isspmatrix_csr(input_exp_mat_norm):
            input_exp_mat_norm = input_exp_mat_norm.tocsr()
        input_exp_mat_norm.sort_indices()
        patches_cells.sort_indices()
        
        diag_data = diag_matrix_sparse.diagonal().astype(np.float32)

        if use_gpu and gpu_enabled:
            if gpu_backend == "torch_sparse":
                import torch
                import warnings
                warnings.filterwarnings('ignore', category=UserWarning)
                
                try:
                    exp_gpu = torch.sparse_csr_tensor(
                        torch.tensor(input_exp_mat_norm.indptr, dtype=torch.int64, device='cuda'),
                        torch.tensor(input_exp_mat_norm.indices, dtype=torch.int64, device='cuda'),
                        torch.tensor(input_exp_mat_norm.data, dtype=torch.float32, device='cuda'),
                        size=input_exp_mat_norm.shape
                    )
                    
                    patches_gpu = torch.sparse_csr_tensor(
                        torch.tensor(patches_cells.indptr, dtype=torch.int64, device='cuda'),
                        torch.tensor(patches_cells.indices, dtype=torch.int64, device='cuda'),
                        torch.tensor(patches_cells.data.astype(np.float32), device='cuda'),
                        size=patches_cells.shape
                    )
                    
                    diag_data_gpu = torch.tensor(diag_data, dtype=torch.float32, device='cuda')
                    
                    patches_dense_gpu = patches_gpu.to_dense()
                    res_dense = torch.sparse.mm(exp_gpu, patches_dense_gpu)
                    res_dense *= diag_data_gpu
                    
                    mean_x = res_dense.mean(dim=1)
                    mean_x2 = (res_dense**2).mean(dim=1)
                    var_gpu = mean_x2 - mean_x**2
                    
                    result_vars = var_gpu.cpu().numpy()
                    
                    del res_dense, patches_dense_gpu, exp_gpu, patches_gpu, diag_data_gpu, var_gpu, mean_x, mean_x2
                    torch.cuda.empty_cache()
                    
                    return np.matrix(result_vars).T
                    
                except Exception as e:
                    print(f"GPU optimization failed, falling back to CPU: {e}")
                    x_kj = input_exp_mat_norm @ (patches_cells @ diag_matrix_sparse)
            else:
                x_kj = input_exp_mat_norm @ (patches_cells @ diag_matrix_sparse)
        else:
            x_kj = input_exp_mat_norm @ (patches_cells @ diag_matrix_sparse)

        # Free up memory
        del patches_cells, patches_cells_centroid, diag_matrix_sparse

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
    """
    Calculates the p-values for genomic data.

    Args:
        input_sp_mat: The input spatial matrix as a numpy array. The dimension is N x D, where N is the number of cells and D is the dimension of coordinates.
        input_exp_mat_raw: The raw expression matrix, which can be a numpy array, pandas DataFrame, or csr_matrix. The dimension is N x P, where N is the number of cells and P is the number of genes.
        d1: Distance threshold 1.
        d2: Distance threshold 2.
        leaf_size: An integer that determines the maximum number of points after which the Ball Tree algorithm opts for a brute-force search approach.
        use_gpu: A boolean value that determines whether to use the GPU.

    Returns:
        A Pandas DataFrame with columns ['gene_names', 'p_values'].
    """

    # Extract column names if input_exp_mat_raw is a Pandas DataFrame, else use indices
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

    # Scale the distance thresholds according to the geometric mean of data spread.
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

    # Calculate p-values
    t_matrix_sum_upper90 = np.quantile(t_matrix_sum, 0.90)
    t_matrix_sum_mid = (val for val in t_matrix_sum if val < t_matrix_sum_upper90)
    log_t_matrix_sum_mid = np.fromiter((np.log(val) for val in t_matrix_sum_mid), dtype=float)
    log_norm_params = (log_t_matrix_sum_mid.mean(), log_t_matrix_sum_mid.std(ddof=1))

    def p_value_generator():
        for val in t_matrix_sum:
            yield 1 - lognorm.cdf(val, scale=np.exp(log_norm_params[0]), s=log_norm_params[1])

    p_values = list(p_value_generator())

    return pd.DataFrame({"gene_names": gene_names, "p_values": p_values})


def combine_p_values(
    list_of_pvalues: List[pd.DataFrame],
    method: str = "fisher"
) -> pd.DataFrame:
    """
    Combines p-values across multiple samples from scBSP using Fisher's or Stouffer's method.
    
    Given the results from multiple samples with gene names and p-values, this
    function merges them by gene and computes a combined p-value for each gene.
    
    Args:
        list_of_pvalues: A list of DataFrames, each with columns 'gene_names' and 'p_values'.
        method: Combination method. One of "fisher" (default) or "stouffer".
    
    Returns:
        A DataFrame with columns:
        - gene_names: The gene identifiers
        - number_samples: Number of datasets contributing to this gene
        - calibrated_p_values: The combined p-value
    
    Raises:
        ValueError: If method is not "fisher" or "stouffer", or if input DataFrames 
                    don't have required columns.
    
    Examples:
        >>> df1 = pd.DataFrame({'gene_names': ['A', 'B', 'C'],
        ...                     'p_values': [0.01, 0.20, 0.03]})
        >>> df2 = pd.DataFrame({'gene_names': ['A', 'C', 'D'],
        ...                     'p_values': [0.04, 0.10, 0.50]})
        >>> df3 = pd.DataFrame({'gene_names': ['B', 'C', 'E'],
        ...                     'p_values': [0.05, 0.02, 0.80]})
        >>> result = combine_p_values([df1, df2, df3], method="fisher")
    """
    
    # Validate method
    if method not in ["fisher", "stouffer"]:
        raise ValueError(f"Method must be 'fisher' or 'stouffer', got '{method}'")
    
    # Validate input DataFrames
    for i, df in enumerate(list_of_pvalues):
        if not isinstance(df, pd.DataFrame):
            raise ValueError(f"Element {i} in list_of_pvalues is not a DataFrame")
        if 'gene_names' not in df.columns or 'p_values' not in df.columns:
            raise ValueError(f"DataFrame {i} must have 'gene_names' and 'p_values' columns")
    
    # If empty list, return empty DataFrame
    if not list_of_pvalues:
        return pd.DataFrame(columns=['gene_names', 'number_samples', 'calibrated_p_values'])
    
    # Rename p_values columns to avoid conflicts during merge
    dfs_renamed = []
    for i, df in enumerate(list_of_pvalues):
        df_copy = df.copy()
        df_copy = df_copy.rename(columns={'p_values': f'p_values_{i+1}'})
        dfs_renamed.append(df_copy)
    
    # Merge all DataFrames on gene_names
    merged = dfs_renamed[0]
    for df in dfs_renamed[1:]:
        merged = pd.merge(merged, df, on='gene_names', how='outer')
    
    # Get p-value columns
    pval_cols = [col for col in merged.columns if col.startswith('p_values_')]
    
    # Calculate combined p-values for each gene
    combined_results = []
    for _, row in merged.iterrows():
        gene_name = row['gene_names']
        pvals = [row[col] for col in pval_cols]
        
        # Filter out NaN values
        valid_pvals = [p for p in pvals if pd.notna(p)]
        k = len(valid_pvals)
        
        if k == 0:
            # No valid p-values for this gene
            combined_results.append({
                'gene_names': gene_name,
                'number_samples': 0,
                'calibrated_p_values': np.nan
            })
            continue
        
        # Apply combination method
        if method == "fisher":
            # Fisher's method: -2 * sum(log(p_i))
            # Avoid log(0) by using a small epsilon
            epsilon = 1e-300
            valid_pvals_safe = [max(p, epsilon) for p in valid_pvals]
            stat = -2 * sum(np.log(valid_pvals_safe))
            # Combined p-value from chi-squared distribution with 2k degrees of freedom
            combined_pval = 1 - chi2.cdf(stat, 2 * k)
            
        elif method == "stouffer":
            # Stouffer's method: sum(Z_i) / sqrt(k)
            # Convert p-values to z-scores
            # Use two-tailed conversion: z = qnorm(1 - p/2) * sign(0.5 - p)
            z_scores = []
            for p in valid_pvals:
                # Avoid extreme values
                p_safe = max(min(p, 1 - 1e-15), 1e-15)
                z = norm.ppf(1 - p_safe/2) * np.sign(0.5 - p_safe)
                z_scores.append(z)
            
            # Combined z-score
            z_combined = sum(z_scores) / np.sqrt(k)
            # Convert back to p-value (two-tailed)
            combined_pval = 2 * (1 - norm.cdf(abs(z_combined)))
        
        combined_results.append({
            'gene_names': gene_name,
            'number_samples': k,
            'calibrated_p_values': combined_pval
        })
    
    # Create result DataFrame
    result_df = pd.DataFrame(combined_results)
    
    # Ensure number_samples is integer type
    result_df['number_samples'] = result_df['number_samples'].astype(int)
    
    return result_df
