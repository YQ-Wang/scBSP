# -*- coding: utf-8 -*-
"""
Created on Mon Nov  6 20:19:23 2023

@author: lijinp yiqingwang
"""


import numpy as np
from scipy.stats import gmean, lognorm
from scipy.sparse import diags, identity, csr_matrix, isspmatrix_csr
from scipy.spatial import KDTree
from sklearn.preprocessing import minmax_scale
from typing import Dict, Tuple, Any


def scale_sparse_minmax(InputExpMatSp: csr_matrix) -> csr_matrix:
    InputExpMatSp_cdx = InputExpMatSp.data

    if len(InputExpMatSp_cdx) / InputExpMatSp.shape[0] / InputExpMatSp.shape[1] > 0.1:
        InputExpMatDen = InputExpMatSp.todense()
        Norm_Exp = minmax_scale(InputExpMatDen,axis=1)
        Created_Spmat = csr_matrix(Norm_Exp)
    else:
        InputExpMatSp_Row = InputExpMatSp.getnnz(axis=0)
        InputExpMatSp_idx = np.r_[0, InputExpMatSp_Row[:-1].cumsum()]
        InputExpMatSp_max = np.maximum.reduceat(InputExpMatSp_cdx, InputExpMatSp_idx)
        InputExpMatSp_min = np.minimum.reduceat(InputExpMatSp_cdx, InputExpMatSp_idx) - 1
        InputExpMatSp_Diff = InputExpMatSp_max - InputExpMatSp_min
        InputExpMatSp_Diff[InputExpMatSp_Diff == 0] = 1  # Prevent division by zero
        InputExpMatSp_Diffs = 1 / InputExpMatSp_Diff
        InputExpMatSp_Diffs = np.repeat(InputExpMatSp_Diffs, InputExpMatSp_Row)
        InputExpMatSp_mins = np.repeat(InputExpMatSp_min, InputExpMatSp_Row)
        InputExpMatSp_vals = (InputExpMatSp_cdx - InputExpMatSp_mins) * InputExpMatSp_Diffs
        rows, cols = InputExpMatSp.nonzero()
        Created_Spmat = csr_matrix((InputExpMatSp_vals, (rows, cols)), shape=InputExpMatSp.shape)
    return(Created_Spmat)


def binary_distance_matrix_Thres(InputSpMatArray: np.ndarray, D_K: float) -> csr_matrix:
    kd_tree = KDTree(InputSpMatArray)
    sparseMat = kd_tree.sparse_distance_matrix(kd_tree, D_K)
    sparseMat[sparseMat > 1] = 1
    return sparseMat + identity(InputSpMatArray.shape[0], format='csr', dtype=sparseMat.dtype)


def spvars(InputcsrMat: csr_matrix, axis: int = None) -> np.ndarray:
    InputcsrMat_squared = InputcsrMat.copy()
    InputcsrMat_squared.data **= 2
    return InputcsrMat_squared.mean(axis) - np.square(InputcsrMat.mean(axis))


def test_scores(InputSpMat: csr_matrix, InputExpMatRaw: csr_matrix, D1: float, D2: float) -> np.ndarray:
    InputExpMatNorm = scale_sparse_minmax(InputExpMatRaw).transpose()
    InputExpMatRaw = InputExpMatRaw.transpose()
    inverted_diag_matrix_cache: Dict[Tuple, csr_matrix] = {}

    def get_inverted_diag_matrix(sum_axis_0: np.ndarray) -> csr_matrix:
        cache_key = tuple(sum_axis_0)
        if cache_key not in inverted_diag_matrix_cache:
            with np.errstate(divide='ignore', invalid='ignore'):
                diag_data = np.reciprocal(sum_axis_0, where=sum_axis_0!=0)
            inverted_diag_matrix_cache[cache_key] = diags(diag_data, offsets=0, format='csr')
        return inverted_diag_matrix_cache[cache_key]

    def var_local_means(DK: float) -> np.ndarray:
        PatchesCells = binary_distance_matrix_Thres(InputSpMat, DK)
        PatchesCells_Centroid = diags((PatchesCells.sum(axis=1) > 1).astype(float).A.ravel(), offsets=0, format='csr')
        PatchesCells -= PatchesCells_Centroid
        sum_axis_0 = PatchesCells.sum(axis=0).A.ravel()
        diag_matrix_sparse = get_inverted_diag_matrix(sum_axis_0)
        X_kj = InputExpMatNorm.dot(PatchesCells).dot(diag_matrix_sparse)
        return spvars(X_kj, axis=1)

    Var_X = np.column_stack([var_local_means(DK).A.ravel() for DK in (D1, D2)])
    Var_X_0_Add = spvars(InputExpMatRaw, axis=1).A.ravel()
    Var_X_0_Add /= max(Var_X_0_Add)
    T_matrix = (Var_X[:, 1] / Var_X[:, 0]) * Var_X_0_Add
    return T_matrix


def granp(InputSpMat: csr_matrix, InputExpMatRaw: Any, D1: float = 1.0, D2: float = 3.0) -> np.ndarray:
    # Normalize patch size
    # Using gmean for geometric mean to scale D1 and D2 accordingly
    scalFactor = gmean(np.quantile(InputSpMat, 0.975, axis=0) - np.quantile(InputSpMat, 0.025, axis=0)) / 0.95 / (InputSpMat.shape[0]) ** (1 / InputSpMat.shape[1])
    D1 *= scalFactor
    D2 *= scalFactor
    
    # Convert expression to sparse matrix if it is not already
    if not isspmatrix_csr(InputExpMatRaw):
        InputExpMatRaw = csr_matrix(InputExpMatRaw)

    # Calculate test scores
    T_matrix_sum = test_scores(InputSpMat, InputExpMatRaw, D1, D2)
    
    # Calculate p-values
    T_matrix_sum_upper90 = np.quantile(T_matrix_sum, 0.90)
    T_matrix_sum_mid = T_matrix_sum[T_matrix_sum < T_matrix_sum_upper90]
    log_T_matrix_sum_mid = np.log(T_matrix_sum_mid)
    LogNormPar = (log_T_matrix_sum_mid.mean(), log_T_matrix_sum_mid.std(ddof=1))
    
    # Vectorized p-value calculation
    pvalues = 1 - lognorm.cdf(T_matrix_sum, scale=np.exp(LogNormPar[0]), s=LogNormPar[1])

    return pvalues
