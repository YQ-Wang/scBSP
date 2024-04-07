import unittest

import numpy as np
import pandas as pd
from scipy.sparse import csr_matrix
from scipy.sparse import random as sparse_random

from scbsp.scbsp import (
    _binary_distance_matrix_threshold,
    _calculate_sparse_variances,
    _get_test_scores,
    _scale_sparse_matrix,
    granp,
)


class TestScaleSparseMinmax(unittest.TestCase):
    def test_scale_sparse_matrix(self):
        # Creating a small sparse matrix with known values
        rows, cols = 3, 3
        data = [4, 2, 1, 4, 5]
        row_indices = [0, 0, 1, 1, 2]
        col_indices = [0, 2, 1, 2, 2]
        test_matrix = csr_matrix((data, (row_indices, col_indices)), shape=(rows, cols))

        # Expected scaled matrix
        # For row 0: max is 4, so 4/4=1 and 2/4=0.5
        # For row 1: max is 4, so 1/4=0.25 and 4/4=1
        # For row 2: only one value which is 5, so it remains 1
        expected_scaled_data = [1, 0.5, 0.25, 1, 1]

        scaled_matrix = _scale_sparse_matrix(test_matrix)

        scaled_data = scaled_matrix.data
        self.assertTrue(
            np.allclose(scaled_data, expected_scaled_data),
            "Scaled data does not match expected values",
        )

    def test_sparse_matrix_scaling(self):
        # Create a sparse matrix with less than 10% non-zero entries
        rows, cols = 10, 10
        density = 0.1
        sparse_matrix = sparse_random(
            rows, cols, density=density, format="csr", dtype=float
        )

        scaled_matrix = _scale_sparse_matrix(sparse_matrix)

        for row in range(scaled_matrix.shape[0]):
            row_data = scaled_matrix[row, :].toarray().flatten()
            if np.any(row_data != 0):
                self.assertEqual(
                    row_data.max(), 1, "Max value in a row should be 1 after scaling"
                )

    def test_empty_matrix(self):
        # Empty matrix
        matrix = csr_matrix((0, 0))

        scaled_matrix = _scale_sparse_matrix(matrix)
        scaled_matrix_dense = np.asarray(scaled_matrix.todense())

        self.assertEqual(scaled_matrix.shape, matrix.shape)
        self.assertIsInstance(scaled_matrix_dense, np.ndarray)


class TestBinaryDistanceMatrixThreshold(unittest.TestCase):
    def test_non_empty_array(self):
        input_array = np.array([[0, 1], [1, 0], [1, 1]])
        d_val = 1.5
        leaf_size = 80

        result = _binary_distance_matrix_threshold(input_array, d_val, leaf_size)

        self.assertIsInstance(result, csr_matrix)
        self.assertEqual(result.shape, (input_array.shape[0], input_array.shape[0]))

    def test_distance_threshold(self):
        input_array = np.array([[0, 0], [3, 3], [6, 6]])
        d_val = 5
        leaf_size = 80

        result = _binary_distance_matrix_threshold(input_array, d_val, leaf_size)

        self.assertIsInstance(result, csr_matrix)
        self.assertTrue((result[result > d_val].count_nonzero()) == 0)


class TestSpvars(unittest.TestCase):
    def test_non_empty_matrix(self):
        # Create a non-empty sparse matrix
        data = np.array([1, 2, 3, 4])
        rows = np.array([0, 0, 1, 1])
        cols = np.array([0, 1, 0, 1])
        matrix = csr_matrix((data, (rows, cols)), shape=(2, 2))

        result = _calculate_sparse_variances(matrix, axis=1)

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

        result = _calculate_sparse_variances(matrix, axis=1)

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
        leaf_size = 80

        result = _get_test_scores(input_sp_mat, input_exp_mat_raw, d1, d2, leaf_size)

        # Check if the result is a numpy.ndarray
        self.assertIsInstance(result, list)
        # Check the shape of the result
        self.assertEqual(len(result), 2)  # Shape depends on your function's logic


class TestGranp(unittest.TestCase):
    def test_p_value_calculation(self):
        input_file = "test/test_data/scenario1_RW1_3-5_1.csv"
        input_date = pd.read_csv(input_file)
        input_sp_mat = input_date[["x", "y", "z"]].to_numpy()
        input_exp_mat_raw = input_date.iloc[:, 3:]

        p_values = granp(input_sp_mat, input_exp_mat_raw)

        self.assertEqual(sum([(i < 0.0001).astype(int) for i in p_values[0:999]]), 996)


if __name__ == "__main__":
    unittest.main()
