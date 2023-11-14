import unittest

import numpy as np
import pandas as pd
from scipy.sparse import csr_matrix, identity, isspmatrix_csr
from scipy.stats import lognorm
from src import (
    binary_distance_matrix_threshold,
    granp,
    scale_sparse_minmax,
    spvars,
    test_scores,
)


class TestScaleSparseMinmax(unittest.TestCase):
    def test_dense_conversion_scaling(self):
        # Sparse matrix with more than 10% non-zero entries
        data = np.array([1, 2, 3, 4, 5, 6, 7, 8, 9, 10])
        rows = np.array([0, 1, 2, 3, 4, 0, 1, 2, 3, 4])
        cols = np.array([0, 0, 0, 0, 0, 1, 1, 1, 1, 1])
        matrix = csr_matrix((data, (rows, cols)), shape=(5, 2))

        scaled_matrix = scale_sparse_minmax(matrix)
        scaled_matrix_dense = np.asarray(scaled_matrix.todense())

        self.assertEqual(scaled_matrix.shape, matrix.shape)
        # Ensure scaled_matrix_dense is ndarray and not np.matrix
        self.assertIsInstance(scaled_matrix_dense, np.ndarray)

    def test_sparse_scaling(self):
        # Sparse matrix with less than 10% non-zero entries
        data = np.array([1, 2, 3])
        rows = np.array([0, 1, 2])
        cols = np.array([0, 0, 0])
        matrix = csr_matrix((data, (rows, cols)), shape=(5, 5))

        scaled_matrix = scale_sparse_minmax(matrix)
        scaled_matrix_dense = np.asarray(scaled_matrix.todense())

        self.assertEqual(scaled_matrix.shape, matrix.shape)
        self.assertIsInstance(scaled_matrix_dense, np.ndarray)

    def test_empty_matrix(self):
        # Empty matrix
        matrix = csr_matrix((0, 0))

        scaled_matrix = scale_sparse_minmax(matrix)
        scaled_matrix_dense = np.asarray(scaled_matrix.todense())

        self.assertEqual(scaled_matrix.shape, matrix.shape)
        self.assertIsInstance(scaled_matrix_dense, np.ndarray)


class TestBinaryDistanceMatrixThreshold(unittest.TestCase):
    def test_non_empty_array(self):
        input_array = np.array([[0, 1], [1, 0], [1, 1]])
        d_val = 1.5

        result = binary_distance_matrix_threshold(input_array, d_val)

        self.assertIsInstance(result, csr_matrix)
        self.assertEqual(result.shape, (input_array.shape[0], input_array.shape[0]))

    def test_distance_threshold(self):
        input_array = np.array([[0, 0], [3, 3], [6, 6]])
        d_val = 5

        result = binary_distance_matrix_threshold(input_array, d_val)

        self.assertIsInstance(result, csr_matrix)
        self.assertTrue((result[result > d_val].count_nonzero()) == 0)


class TestSpvars(unittest.TestCase):
    def test_non_empty_matrix(self):
        # Create a non-empty sparse matrix
        data = np.array([1, 2, 3, 4])
        rows = np.array([0, 0, 1, 1])
        cols = np.array([0, 1, 0, 1])
        matrix = csr_matrix((data, (rows, cols)), shape=(2, 2))

        result = spvars(matrix, axis=1)

        # Check if the result is a numpy.ndarray
        self.assertIsInstance(result, np.ndarray)
        # Check the shape of the result
        self.assertEqual(result.shape, (2, 1))  # One value per row

    def test_known_data(self):
        # Create a sparse matrix with known data
        data = np.array([1, 2, 2, 4])
        rows = np.array([0, 0, 1, 1])
        cols = np.array([0, 1, 0, 1])
        matrix = csr_matrix((data, (rows, cols)), shape=(2, 2))

        result = spvars(matrix, axis=1)

        expected = np.array([[0.25], [1.0]])

        # Check if the result matches the expected result
        np.testing.assert_array_almost_equal(result, expected)


class TestTestScores(unittest.TestCase):
    def test_non_empty_matrices(self):
        # Create non-empty numpy array and csr_matrix
        input_sp_mat = np.array([[0, 1], [1, 0], [1, 1]])
        data = np.array([1, 2, 3, 4, 5, 6])
        rows = np.array([0, 1, 2, 0, 1, 2])
        cols = np.array([0, 0, 0, 1, 1, 1])
        input_exp_mat_raw = csr_matrix((data, (rows, cols)), shape=(3, 2))

        # Define d1 and d2
        d1 = 1.0
        d2 = 3.0

        result = test_scores(input_sp_mat, input_exp_mat_raw, d1, d2)

        # Check if the result is a numpy.ndarray
        self.assertIsInstance(result, np.ndarray)
        # Check the shape of the result
        self.assertEqual(result.shape, (2,))  # Shape depends on your function's logic


class TestGranp(unittest.TestCase):
    def test_p_value_calculation(self):
        input_file = "test/test_data/scenario1_RW1_3-5_1.csv"
        input_date = pd.read_csv(input_file)
        input_sp_mat = input_date[["x", "y", "z"]].to_numpy()
        input_exp_mat_raw = input_date.iloc[:, 3:]

        p_values = granp(input_sp_mat, input_exp_mat_raw)

        self.assertEqual(sum([(i < 0.0001).astype(int) for i in p_values[0:999]]), 998)
        self.assertEqual(
            sum([(i < 0.0001).astype(int) for i in p_values[1000:9999]]), 1
        )


if __name__ == "__main__":
    unittest.main()
