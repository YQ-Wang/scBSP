"""
@author: lijinp yiqingwang

This module utilizes a granularity-based dimension-agnostic tool, single-cell
big-small patch (scBSP), implementing sparse matrix operation and HNSW method
for distance calculation, for the identification of spatially variable genes
on large-scale data.
"""

from typing import List, Tuple, Union

import numpy as np
import pandas as pd  # type: ignore
from scipy.sparse import csr_matrix, diags, identity, isspmatrix_csr  # type: ignore
from scipy.stats import gmean, lognorm

try:
    import hnswlib

    use_hnsw = True
except ImportError:
    from sklearn.neighbors import BallTree

    use_hnsw = False


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

    if use_hnsw is True:
        labels, distances = _query_hnsw_index(input_sparse_mat_array)
        within_d_val = distances <= d_val**2
        rows = np.repeat(np.arange(labels.shape[0]), labels.shape[1])[
            within_d_val.flatten()
        ]
        cols = labels.flatten()[within_d_val.flatten()]
    else:
        ball_tree = BallTree(input_sparse_mat_array, leaf_size=leaf_size)
        indices = ball_tree.query_radius(
            input_sparse_mat_array, r=d_val, return_distance=False
        )
        rows = np.repeat(
            np.arange(input_sparse_mat_array.shape[0]), [len(i) for i in indices]
        )
        cols = np.concatenate(indices)

    # Construct binary csr_matrix
    data = np.ones_like(rows)

    sparse_mat = csr_matrix(
        (data, (rows, cols)),
        shape=(input_sparse_mat_array.shape[0], input_sparse_mat_array.shape[0]),
    )
    sparse_mat += identity(input_sparse_mat_array.shape[0], format="csr", dtype=bool)

    return sparse_mat


def _calculate_sparse_variances(input_csr_mat: csr_matrix, axis: int) -> List[float]:
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


def _query_hnsw_index(input_sp_mat: csr_matrix) -> Tuple[np.ndarray, np.ndarray]:
    """
    Queries an HNSW index with a sparse matrix to find k-nearest neighbors.

    Args:
        input_sp_mat: The input sparse matrix.

    Returns:
        A tuple containing the labels and distances of the k-nearest neighbors.
    """

    hnsw_index = hnswlib.Index(space="l2", dim=input_sp_mat.shape[1])
    hnsw_index.init_index(max_elements=input_sp_mat.shape[0], ef_construction=200, M=16)
    hnsw_index.add_items(input_sp_mat)
    hnsw_index.set_ef(100)  # ef should always be > k
    labels, distances = hnsw_index.knn_query(
        input_sp_mat, k=min(80, input_sp_mat.shape[0])
    )

    return labels, distances


def _get_test_scores(
    input_sp_mat: np.ndarray,
    input_exp_mat_raw: csr_matrix,
    d1: float,
    d2: float,
    leaf_size: int,
) -> List[float]:
    """
    Calculates test scores for genomic data based on input sparse matrices and distance thresholds.

    Args:
        input_sp_mat: The input spatial matrix as a numpy array.
        input_exp_mat_raw: The raw expression matrix in csr_matrix format.
        d1: Distance threshold 1.
        d2: Distance threshold 2.
        leaf_size: An integer that determines the maximum number of points after which the Ball Tree algorithm opts for a brute-force search approach.

    Returns:
        A list of test scores.
    """

    input_exp_mat_norm = _scale_sparse_matrix(input_exp_mat_raw).transpose()
    input_exp_mat_raw = input_exp_mat_raw.transpose()

    def _get_inverted_diag_matrix(sum_axis_0: np.ndarray) -> csr_matrix:
        with np.errstate(divide="ignore", invalid="ignore"):
            diag_data = np.reciprocal(sum_axis_0, where=sum_axis_0 != 0)
        return diags(diag_data, offsets=0, format="csr")

    def _var_local_means(
        input_sp_mat: csr_matrix,
        d_val: float,
        input_exp_mat_norm: csr_matrix,
        leaf_size: int,
    ) -> List[float]:
        patches_cells = _binary_distance_matrix_threshold(
            input_sp_mat, d_val, leaf_size
        )
        patches_cells_centroid = diags(
            (patches_cells.sum(axis=1) > 1).astype(float).A.ravel(),
            offsets=0,
            format="csr",
        )
        patches_cells -= patches_cells_centroid
        sum_axis_0 = patches_cells.sum(axis=0).A.ravel()
        diag_matrix_sparse = _get_inverted_diag_matrix(sum_axis_0)
        x_kj = input_exp_mat_norm.dot(patches_cells.dot(diag_matrix_sparse))
        return _calculate_sparse_variances(x_kj, axis=1)

    var_x = np.column_stack([_var_local_means(input_sp_mat, d_val, input_exp_mat_norm, leaf_size).A.ravel() for d_val in (d1, d2)])  # type: ignore
    var_x_0_add = _calculate_sparse_variances(input_exp_mat_raw, axis=1).A.ravel()  # type: ignore
    var_x_0_add /= max(var_x_0_add)
    t_matrix = (var_x[:, 1] / var_x[:, 0]) * var_x_0_add
    return t_matrix.tolist()


def granp(
    input_sp_mat: np.ndarray,
    input_exp_mat_raw: Union[np.ndarray, pd.DataFrame, csr_matrix],
    d1: float = 1.0,
    d2: float = 3.0,
    leaf_size: int = 80,
) -> List[float]:
    """
    Calculates the p-values for genomic data.

    Args:
        input_sp_mat: The input spatial matrix as a numpy array. The dimension is N x D, where N is the number of cells and D is the dimension of coordinates.
        input_exp_mat_raw: The raw expression matrix, which can be a numpy array, pandas DataFrame, or csr_matrix. The dimension is N x P, where N is the number of cells and P is the number of genes.
        d1: Distance threshold 1.
        d2: Distance threshold 2.
        leaf_size: An integer that determines the maximum number of points after which the Ball Tree algorithm opts for a brute-force search approach.

    Returns:
        A list of p-values.
    """

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

    # Ensure the expression matrix is in csr_matrix format.
    if not isspmatrix_csr(input_exp_mat_raw):
        input_exp_mat_raw = csr_matrix(input_exp_mat_raw)

    t_matrix_sum = _get_test_scores(input_sp_mat, input_exp_mat_raw, d1, d2, leaf_size)

    # Calculate p-values
    t_matrix_sum_upper90 = np.quantile(t_matrix_sum, 0.90)
    t_matrix_sum_mid = [val for val in t_matrix_sum if val < t_matrix_sum_upper90]
    log_t_matrix_sum_mid = np.log(t_matrix_sum_mid)
    log_norm_params = (log_t_matrix_sum_mid.mean(), log_t_matrix_sum_mid.std(ddof=1))

    # Calculate p-values using the log-normal distribution.
    p_values = 1 - lognorm.cdf(
        t_matrix_sum, scale=np.exp(log_norm_params[0]), s=log_norm_params[1]
    )

    return p_values
