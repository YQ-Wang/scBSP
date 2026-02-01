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


try:
    import torch   # type: ignore
    if not torch.cuda.is_available():
        gpu_enabled = False
        print("CUDA is not available, using CPU instead.")
except ImportError:
    gpu_enabled = False

def _scipy_to_torch_sparse(mat: csr_matrix) -> "torch.Tensor":
    """Helper to convert scipy CSR to torch sparse CSR."""
    if not gpu_enabled:
        raise RuntimeError("GPU not enabled")
    mat = mat.astype(np.float32)
    crow = torch.tensor(mat.indptr, dtype=torch.int64, device="cuda")
    col = torch.tensor(mat.indices, dtype=torch.int64, device="cuda")
    val = torch.tensor(mat.data, dtype=torch.float32, device="cuda")
    return torch.sparse_csr_tensor(crow, col, val, size=mat.shape)


def _gpu_sparse_matmul(
    sparse_a: Union[csr_matrix, "torch.Tensor"], 
    sparse_b: Union[csr_matrix, "torch.Tensor"]
) -> csr_matrix:
    """
    Performs sparse matrix multiplication on GPU using PyTorch sparse CSR tensors.
    
    Args:
        sparse_a: Left matrix (csr_matrix or torch.Tensor).
        sparse_b: Right matrix (csr_matrix or torch.Tensor).
    
    Returns:
        Result as a scipy csr_matrix.
    """
    if not gpu_enabled:
        # Fallback if somehow called without GPU, but inputs must be scipy
        if not isinstance(sparse_a, csr_matrix) or not isinstance(sparse_b, csr_matrix):
            raise ValueError("GPU disabled but inputs are not scipy matrices")
        return sparse_a @ sparse_b
    
    # Convert to torch if needed
    a_torch = sparse_a if isinstance(sparse_a, torch.Tensor) else _scipy_to_torch_sparse(sparse_a)
    b_torch = sparse_b if isinstance(sparse_b, torch.Tensor) else _scipy_to_torch_sparse(sparse_b)
    
    # Perform sparse matmul on GPU
    result = torch.sparse.mm(a_torch, b_torch)
    
    # Convert back to scipy CSR
    if result.is_sparse_csr:
        result_cpu = result.cpu()
        result_csr = csr_matrix(
            (result_cpu.values().numpy(), 
             result_cpu.col_indices().numpy(), 
             result_cpu.crow_indices().numpy()),
            shape=result.shape
        )
    else:
        # Fallback: convert dense result back to sparse
        result_csr = csr_matrix(result.cpu().numpy())
    
    # Free GPU memory (only if we created new tensors)
    if not isinstance(sparse_a, torch.Tensor):
        del a_torch
    if not isinstance(sparse_b, torch.Tensor):
        del b_torch
    del result
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    
    return result_csr


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

    # Vectorized row max computation using reduceat
    non_empty_mask = row_indices > 0
    row_max = np.ones(input_exp_mat.shape[0], dtype=data.dtype)

    if non_empty_mask.any():
        # Get starting indices only for non-empty rows
        non_empty_starts = row_idx[:-1][non_empty_mask]
        row_max[non_empty_mask] = np.maximum.reduceat(data, non_empty_starts)

    # Scale the data based on the row max
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
    return _binary_distance_matrix_with_tree(ball_tree, input_sparse_mat_array, d_val)


def _binary_distance_matrix_with_tree(
    ball_tree: BallTree, input_sparse_mat_array: np.ndarray, d_val: float
) -> csr_matrix:
    """
    Creates a binary distance matrix using a pre-built BallTree.

    Args:
        ball_tree: Pre-built BallTree for the input points.
        input_sparse_mat_array: The input sparse matrix array.
        d_val: The distance threshold.

    Returns:
        A csr_matrix representing the binary distance matrix.
    """
    indices = ball_tree.query_radius(
        input_sparse_mat_array, r=d_val, return_distance=False
    )

    # Vectorized CSR construction - avoid generator overhead
    lengths = np.array([len(idx) for idx in indices])
    total_nnz = lengths.sum()

    # Allocate arrays directly using numpy operations
    rows = np.repeat(np.arange(len(indices)), lengths)
    cols = np.concatenate(indices) if total_nnz > 0 else np.array([], dtype=np.intp)
    data = np.ones(total_nnz, dtype=np.int8)

    sparse_mat = csr_matrix(
        (data, (rows, cols)),
        shape=(input_sparse_mat_array.shape[0], input_sparse_mat_array.shape[0]),
        dtype=np.int8
    )

    return sparse_mat


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

    ball_tree = BallTree(input_sp_mat, leaf_size=leaf_size)

    input_exp_mat_norm_gpu = None
    if use_gpu and gpu_enabled:
        try:
            input_exp_mat_norm_gpu = _scipy_to_torch_sparse(input_exp_mat_norm)
        except Exception as e:
            print(f"Warning: GPU conversion failed: {e}. Falling back to CPU/per-call conversion.")
            use_gpu = False 

    def _var_local_means(
        ball_tree: BallTree,
        input_sp_mat: np.ndarray,
        d_val: float,
        input_exp_mat_norm: csr_matrix,
        use_gpu: bool,
        input_exp_mat_norm_gpu: Optional["torch.Tensor"] = None
    ) -> np.matrix:
        patches_cells = _binary_distance_matrix_with_tree(
            ball_tree, input_sp_mat, d_val
        )
        
        sum_axis_0 = patches_cells.sum(axis=0).A.ravel().astype(np.float64)

        inv_sum = np.reciprocal(sum_axis_0, where=sum_axis_0 != 0, out=np.zeros_like(sum_axis_0, dtype=np.float64))
        patches_scaled = patches_cells.astype(np.float64).multiply(inv_sum)

        if use_gpu and gpu_enabled:
            n_genes = input_exp_mat_norm.shape[0]
            batch_size = 5000 
            
            lhs = input_exp_mat_norm_gpu if input_exp_mat_norm_gpu is not None else input_exp_mat_norm

            if n_genes <= batch_size:
                x_kj = _gpu_sparse_matmul(lhs, patches_scaled)
            else:
                results = []
                for i in range(0, n_genes, batch_size):
                    # For batching, we need slicing. 
                    # If lhs is tensor, slicing sparse tensors in PyTorch can be tricky/limited.
                    # Safe approach: pass the scipy batch and let _gpu_sparse_matmul convert it, 
                    # OR slice the tensor if supported. 
                    # PyTorch sparse slicing is not fully robust yet in all versions.
                    # Fallback to scipy slicing + conversion for batches to be safe.
                    
                    if input_exp_mat_norm_gpu is not None:
                         # Slicing sparse tensors is not directly supported in older torch versions.
                         # We'll rely on the original matrix for slicing to be safe.
                         batch = input_exp_mat_norm[i:i+batch_size]
                         batch_result = _gpu_sparse_matmul(batch, patches_scaled)
                    else:
                         batch = input_exp_mat_norm[i:i+batch_size]
                         batch_result = _gpu_sparse_matmul(batch, patches_scaled)
                    
                    results.append(batch_result)
                x_kj = scipy.sparse.vstack(results, format='csr')
        else:
            x_kj = input_exp_mat_norm @ patches_scaled

        # Free up memory
        del patches_cells, patches_scaled

        return _calculate_sparse_variances(x_kj, axis=1)

    def var_x_generator():
        for d_val in (d1, d2):
            yield _var_local_means(
                ball_tree, input_sp_mat, d_val, input_exp_mat_norm, use_gpu, input_exp_mat_norm_gpu
            ).A.ravel()

    var_x = np.column_stack(list(var_x_generator()))
    
    # Validation check: if gpu was used, we should clear the large tensor
    if input_exp_mat_norm_gpu is not None:
        del input_exp_mat_norm_gpu
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            
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

    # Calculate p-values - vectorized implementation
    t_matrix_array = np.asarray(t_matrix_sum)
    t_matrix_sum_upper90 = np.quantile(t_matrix_array, 0.90)

    # Vectorized filtering and log computation
    mask = t_matrix_array < t_matrix_sum_upper90
    log_t_matrix_sum_mid = np.log(t_matrix_array[mask])
    log_norm_params = (log_t_matrix_sum_mid.mean(), log_t_matrix_sum_mid.std(ddof=1))

    # Vectorized p-value computation - single CDF call on entire array
    p_values = 1 - lognorm.cdf(t_matrix_array, scale=np.exp(log_norm_params[0]), s=log_norm_params[1])
    p_values = p_values.tolist()

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
    
    # Vectorized p-value combination - avoid iterrows()
    gene_names = merged['gene_names'].to_numpy()
    pval_matrix = merged[pval_cols].to_numpy()

    # Count non-NaN values per row (vectorized)
    valid_counts = np.sum(~np.isnan(pval_matrix), axis=1)

    # Initialize result arrays
    n_genes = len(gene_names)
    combined_pvals = np.full(n_genes, np.nan)

    # Process genes with at least one valid p-value
    valid_mask = valid_counts > 0

    if valid_mask.any():
        if method == "fisher":
            # Fisher's method: -2 * sum(log(p_i))
            epsilon = 1e-300
            pval_safe = np.maximum(pval_matrix, epsilon)
            # Replace NaN with 1 (log(1)=0, won't affect sum)
            pval_safe = np.where(np.isnan(pval_matrix), 1.0, pval_safe)
            log_pvals = np.log(pval_safe)
            stat = -2 * np.sum(log_pvals, axis=1)
            # Combined p-value from chi-squared distribution with 2k degrees of freedom
            combined_pvals[valid_mask] = 1 - chi2.cdf(
                stat[valid_mask], 2 * valid_counts[valid_mask]
            )

        elif method == "stouffer":
            # Stouffer's method: sum(Z_i) / sqrt(k)
            # Clip p-values to avoid extreme z-scores
            pval_clipped = np.clip(pval_matrix, 1e-15, 1 - 1e-15)
            # Convert to z-scores (two-tailed)
            z_scores = norm.ppf(1 - pval_clipped / 2) * np.sign(0.5 - pval_clipped)
            # Replace NaN with 0 (won't affect sum)
            z_scores = np.where(np.isnan(pval_matrix), 0, z_scores)
            # Combined z-score per gene
            z_combined = np.sum(z_scores, axis=1) / np.sqrt(np.maximum(valid_counts, 1))
            # Convert back to p-value (two-tailed)
            combined_pvals[valid_mask] = 2 * (1 - norm.cdf(np.abs(z_combined[valid_mask])))

    # Create result DataFrame directly from arrays
    result_df = pd.DataFrame({
        'gene_names': gene_names,
        'number_samples': valid_counts.astype(int),
        'calibrated_p_values': combined_pvals
    })
    
    return result_df
