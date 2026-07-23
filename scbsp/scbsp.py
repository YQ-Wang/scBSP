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
gpu_backend: Optional[str] = "torch_sparse"

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

    output_dtype = (
        input_exp_mat.data.dtype
        if np.issubdtype(input_exp_mat.data.dtype, np.inexact)
        else np.dtype(np.float64)
    )
    scaled_matrix = input_exp_mat.astype(output_dtype, copy=True)

    row_lengths = np.diff(scaled_matrix.indptr)
    nnz_starts = scaled_matrix.indptr[:-1]

    non_empty_mask = row_lengths > 0
    row_max = np.ones(scaled_matrix.shape[0], dtype=scaled_matrix.data.dtype)

    if non_empty_mask.any():
        row_max[non_empty_mask] = np.maximum.reduceat(
            scaled_matrix.data, nnz_starts[non_empty_mask]
        )

    row_divisors = np.repeat(row_max, row_lengths)
    np.divide(
        scaled_matrix.data,
        row_divisors,
        out=scaled_matrix.data,
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
    patches_by_radius = [
        _binary_distance_matrix_threshold(
            input_sp_mat, d_val, leaf_size, ball_tree=ball_tree
        )
        for d_val in (d1, d2)
    ]

    def _cpu_local_variances() -> List[np.ndarray]:
        # The neighbor matrices are sparse, but local means are generally dense.
        # Keep both scaled matrices so each dense expression batch can be reused
        # for both radii instead of being materialized twice.
        patches_scaled = []
        for patches_cells in patches_by_radius:
            col_counts = np.asarray(
                patches_cells.getnnz(axis=0), dtype=np.float64
            )
            inv_sum = np.reciprocal(
                col_counts,
                where=col_counts != 0,
                out=np.zeros_like(col_counts),
            )
            patches_scaled.append(patches_cells.multiply(inv_sum).tocsr())

        n_genes, n_cells = input_exp_mat_norm.shape
        batch_size = max(1, 10_000_000 // n_cells)
        result_vars: List[List[np.ndarray]] = [
            [] for _ in patches_scaled
        ]

        for start_idx in range(0, n_genes, batch_size):
            end_idx = min(start_idx + batch_size, n_genes)
            exp_batch = input_exp_mat_norm[start_idx:end_idx, :].toarray()

            for radius_idx, patches_matrix in enumerate(patches_scaled):
                x_kj_batch = np.asarray(exp_batch @ patches_matrix)
                result_vars[radius_idx].append(x_kj_batch.var(axis=1))

        return [
            np.concatenate(radius_vars) if radius_vars else np.array([])
            for radius_vars in result_vars
        ]

    def _gpu_batch_size(n_genes: int, n_cells: int) -> int:
        # A sparse.mm batch temporarily holds both its float32 input and output.
        # Retain the previous 250M-element ceiling, but reduce it when current
        # free VRAM cannot safely support those two dense tensors.
        max_dense_elements = 250_000_000
        try:
            free_bytes, _ = torch.cuda.mem_get_info()
            reserve_bytes = 256 * 1024**2
            usable_bytes = max(0, free_bytes - reserve_bytes)
            dense_bytes = int(usable_bytes * 0.60)
            bytes_per_element = 2 * np.dtype(np.float32).itemsize
            memory_limited_elements = max(1, dense_bytes // bytes_per_element)
            max_dense_elements = min(
                max_dense_elements, memory_limited_elements
            )
        except (AttributeError, RuntimeError):
            # Older torch versions may not expose mem_get_info. The established
            # element ceiling remains a safe fallback for supported GPUs.
            pass

        return max(1, min(n_genes, max_dense_elements // n_cells))

    def _gpu_local_variances() -> List[np.ndarray]:
        import warnings

        with warnings.catch_warnings():
            warnings.filterwarnings("ignore", category=UserWarning)

            patches_gpu = []
            inverse_counts_gpu = []
            for patches_cells in patches_by_radius:
                col_counts = np.asarray(
                    patches_cells.getnnz(axis=0), dtype=np.float32
                )
                inv_sum = np.reciprocal(
                    col_counts,
                    where=col_counts != 0,
                    out=np.zeros_like(col_counts),
                )
                patches_t_csr = patches_cells.T.tocsr()
                patches_gpu.append(
                    torch.sparse_csr_tensor(
                        torch.from_numpy(
                            patches_t_csr.indptr.astype(np.int64)
                        ).to("cuda"),
                        torch.from_numpy(
                            patches_t_csr.indices.astype(np.int64)
                        ).to("cuda"),
                        torch.from_numpy(
                            patches_t_csr.data.astype(np.float32)
                        ).to("cuda"),
                        size=patches_t_csr.shape,
                    )
                )
                inverse_counts_gpu.append(
                    torch.from_numpy(inv_sum).to("cuda").view(-1, 1)
                )

            n_genes, n_cells = input_exp_mat_norm.shape
            batch_size = _gpu_batch_size(n_genes, n_cells)
            result_vars: List[List[np.ndarray]] = [
                [] for _ in patches_gpu
            ]

            def _process_batch(
                start_idx: int, end_idx: int
            ) -> List[np.ndarray]:
                exp_batch_np = (
                    input_exp_mat_norm[start_idx:end_idx, :]
                    .toarray()
                    .T.astype(np.float32)
                )
                exp_batch_gpu = torch.from_numpy(exp_batch_np).to("cuda")
                batch_results = []

                for patches_matrix, inv_sum_gpu in zip(
                    patches_gpu, inverse_counts_gpu
                ):
                    res_batch_gpu = torch.sparse.mm(
                        patches_matrix, exp_batch_gpu
                    )
                    res_batch_gpu *= inv_sum_gpu

                    # Preserve the existing population-variance calculation.
                    mean_x = res_batch_gpu.mean(dim=0)
                    res_batch_gpu -= mean_x
                    res_batch_gpu.square_()
                    var_batch_gpu = res_batch_gpu.mean(dim=0)
                    batch_results.append(var_batch_gpu.cpu().numpy())
                    del res_batch_gpu, mean_x, var_batch_gpu

                return batch_results

            start_idx = 0
            while start_idx < n_genes:
                end_idx = min(start_idx + batch_size, n_genes)
                try:
                    batch_results = _process_batch(start_idx, end_idx)
                except RuntimeError as exc:
                    oom_type = getattr(
                        torch.cuda, "OutOfMemoryError", ()
                    )
                    is_cuda_oom = (
                        isinstance(oom_type, type)
                        and isinstance(exc, oom_type)
                    ) or "out of memory" in str(exc).lower()
                    if not is_cuda_oom or end_idx - start_idx == 1:
                        raise

                    batch_size = max(1, (end_idx - start_idx) // 2)
                    torch.cuda.empty_cache()
                    continue

                for radius_idx, radius_result in enumerate(batch_results):
                    result_vars[radius_idx].append(radius_result)
                start_idx = end_idx

            return [
                np.concatenate(radius_vars) if radius_vars else np.array([])
                for radius_vars in result_vars
            ]

    if use_gpu and gpu_enabled:
        try:
            local_variances = _gpu_local_variances()
        except Exception as exc:
            print(f"GPU optimization failed, falling back to CPU: {exc}")
            local_variances = _cpu_local_variances()
    else:
        local_variances = _cpu_local_variances()

    var_x = np.column_stack(local_variances)
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
