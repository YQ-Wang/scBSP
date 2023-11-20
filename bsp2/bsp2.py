"""
Created on Mon Nov  6 20:19:23 2023

@author: lijinp yiqingwang
"""


from typing import Dict, List, Tuple, Union

import numpy as np
import pandas as pd  # type: ignore
from scipy.sparse import (csr_matrix, diags, identity,  # type: ignore
                          isspmatrix_csr)
from scipy.spatial import KDTree  # type: ignore
from scipy.stats import gmean, lognorm  # type: ignore
from sklearn.preprocessing import minmax_scale  # type: ignore


def _scale_sparse_minmax(input_exp_mat: csr_matrix) -> csr_matrix:
    if input_exp_mat.shape[0] == 0 or input_exp_mat.shape[1] == 0:
        return input_exp_mat

    input_exp_mat_cdx = input_exp_mat.data

    if len(input_exp_mat_cdx) / input_exp_mat.shape[0] / input_exp_mat.shape[1] > 0.1:
        input_exp_mat_den = input_exp_mat.todense()
        input_exp_mat_den_array = np.asarray(input_exp_mat_den)
        norm_exp = minmax_scale(input_exp_mat_den_array, axis=1)
        created_sparse_mat = csr_matrix(norm_exp)
    else:
        input_exp_mat_row = input_exp_mat.getnnz(axis=0)
        input_exp_mat_idx = np.r_[0, input_exp_mat_row[:-1].cumsum()]
        input_exp_mat_max = np.maximum.reduceat(input_exp_mat_cdx, input_exp_mat_idx)
        input_exp_mat_min = (
            np.minimum.reduceat(input_exp_mat_cdx, input_exp_mat_idx) - 1
        )
        input_exp_mat_diff = input_exp_mat_max - input_exp_mat_min
        input_exp_mat_diff[input_exp_mat_diff == 0] = 1  # Prevent division by zero
        input_exp_mat_diffs = 1 / input_exp_mat_diff
        input_exp_mat_diffs = np.repeat(input_exp_mat_diffs, input_exp_mat_row)
        input_exp_mat_mins = np.repeat(input_exp_mat_min, input_exp_mat_row)
        input_exp_mat_vals = (
            input_exp_mat_cdx - input_exp_mat_mins
        ) * input_exp_mat_diffs
        rows, cols = input_exp_mat.nonzero()
        created_sparse_mat = csr_matrix(
            (input_exp_mat_vals, (rows, cols)), shape=input_exp_mat.shape
        )
    return created_sparse_mat


def _binary_distance_matrix_threshold(
    input_sparse_mat_array: np.ndarray, d_val: float
) -> csr_matrix:
    kd_tree = KDTree(input_sparse_mat_array)
    sparse_mat = kd_tree.sparse_distance_matrix(kd_tree, d_val)

    if not isspmatrix_csr(sparse_mat):
        sparse_mat = csr_matrix(sparse_mat)

    sparse_mat[sparse_mat > 1] = 1
    return sparse_mat + identity(
        input_sparse_mat_array.shape[0], format="csr", dtype=sparse_mat.dtype
    )


def _spvars(input_csr_mat: csr_matrix, axis: int) -> List[float]:
    input_csr_mat_squared = input_csr_mat.copy()
    input_csr_mat_squared.data **= 2
    return input_csr_mat_squared.mean(axis) - np.square(input_csr_mat.mean(axis))


def _test_scores(
    input_sp_mat: np.ndarray, input_exp_mat_raw: csr_matrix, d1: float, d2: float
) -> List[float]:
    input_exp_mat_norm = _scale_sparse_minmax(input_exp_mat_raw).transpose()
    input_exp_mat_raw = input_exp_mat_raw.transpose()
    inverted_diag_matrix_cache: Dict[Tuple, csr_matrix] = {}

    def _get_inverted_diag_matrix(sum_axis_0: np.ndarray) -> csr_matrix:
        cache_key = tuple(sum_axis_0)
        if cache_key not in inverted_diag_matrix_cache:
            with np.errstate(divide="ignore", invalid="ignore"):
                diag_data = np.reciprocal(sum_axis_0, where=sum_axis_0 != 0)
            inverted_diag_matrix_cache[cache_key] = diags(
                diag_data, offsets=0, format="csr"
            )
        return inverted_diag_matrix_cache[cache_key]

    def _var_local_means(d_val: float) -> list:
        patches_cells = _binary_distance_matrix_threshold(input_sp_mat, d_val)
        patches_cells_centroid = diags(
            (patches_cells.sum(axis=1) > 1).astype(float).A.ravel(),
            offsets=0,
            format="csr",
        )
        patches_cells -= patches_cells_centroid
        sum_axis_0 = patches_cells.sum(axis=0).A.ravel()
        diag_matrix_sparse = _get_inverted_diag_matrix(sum_axis_0)
        x_kj = input_exp_mat_norm.dot(patches_cells).dot(diag_matrix_sparse)
        return _spvars(x_kj, axis=1)

    var_x = np.column_stack([_var_local_means(d_val).A.ravel() for d_val in (d1, d2)])  # type: ignore
    var_x_0_add = _spvars(input_exp_mat_raw, axis=1).A.ravel()  # type: ignore
    var_x_0_add /= max(var_x_0_add)
    t_matrix = (var_x[:, 1] / var_x[:, 0]) * var_x_0_add
    return t_matrix.tolist()


def granp(
    input_sp_mat: np.ndarray,
    input_exp_mat_raw: Union[np.ndarray, pd.DataFrame, csr_matrix],
    d1: float = 1.0,
    d2: float = 3.0,
) -> List[float]:
    # Normalize patch size
    # Using gmean for geometric mean to scale d1 and d2 accordingly
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

    # Convert expression to sparse matrix if it is not already
    if not isspmatrix_csr(input_exp_mat_raw):
        input_exp_mat_raw = csr_matrix(input_exp_mat_raw)

    # Calculate test scores
    t_matrix_sum = _test_scores(input_sp_mat, input_exp_mat_raw, d1, d2)

    # Calculate p-values
    t_matrix_sum_upper90 = np.quantile(t_matrix_sum, 0.90)
    t_matrix_sum_mid = [val for val in t_matrix_sum if val < t_matrix_sum_upper90]
    log_t_matrix_sum_mid = np.log(t_matrix_sum_mid)
    log_norm_params = (log_t_matrix_sum_mid.mean(), log_t_matrix_sum_mid.std(ddof=1))

    # Vectorized p-value calculation
    p_values = 1 - lognorm.cdf(
        t_matrix_sum, scale=np.exp(log_norm_params[0]), s=log_norm_params[1]
    )

    return p_values
