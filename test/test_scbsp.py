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
    combine_p_values,
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
        use_gpu = False

        result = _get_test_scores(input_sp_mat, input_exp_mat_raw, d1, d2, leaf_size, use_gpu)

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

        self.assertEqual((p_values['p_values'].iloc[0:999] < 0.0001).sum(), 996)


class TestGranpAndCombinePValues(unittest.TestCase):
    def test_integration_granp_and_combine_p_values(self):
        input_file = "test/test_data/scenario1_RW1_3-5_1.csv"
        input_data = pd.read_csv(input_file)

        input_sp_mat = input_data[["x", "y", "z"]].to_numpy()
        input_exp_mat_raw = input_data.iloc[:, 3:]
        
        # Simulate having 3 different samples/conditions by subsetting the data
        n_cells = input_sp_mat.shape[0]
        sample1_size = n_cells // 3
        sample2_size = n_cells // 3
        sample3_size = n_cells - sample1_size - sample2_size
        
        # Create three samples with overlapping genes but different cells
        sample1_sp = input_sp_mat[:sample1_size]
        sample1_exp = input_exp_mat_raw.iloc[:sample1_size]
        
        sample2_sp = input_sp_mat[sample1_size:sample1_size+sample2_size]
        sample2_exp = input_exp_mat_raw.iloc[sample1_size:sample1_size+sample2_size]
        
        sample3_sp = input_sp_mat[sample1_size+sample2_size:]
        sample3_exp = input_exp_mat_raw.iloc[sample1_size+sample2_size:]
        
        # Run granp on each sample
        print("Running granp on sample 1...")
        p_values_sample1 = granp(sample1_sp, sample1_exp, d1=1.0, d2=3.0)
        self.assertIsInstance(p_values_sample1, pd.DataFrame)
        self.assertIn('gene_names', p_values_sample1.columns)
        self.assertIn('p_values', p_values_sample1.columns)
        
        print("Running granp on sample 2...")
        p_values_sample2 = granp(sample2_sp, sample2_exp, d1=1.0, d2=3.0)
        self.assertIsInstance(p_values_sample2, pd.DataFrame)
        
        print("Running granp on sample 3...")
        p_values_sample3 = granp(sample3_sp, sample3_exp, d1=1.0, d2=3.0)
        self.assertIsInstance(p_values_sample3, pd.DataFrame)
        
        # Combine p-values using Fisher's method
        print("Combining p-values using Fisher's method...")
        combined_fisher = combine_p_values(
            [p_values_sample1, p_values_sample2, p_values_sample3],
            method="fisher"
        )
        
        # Validate combined results
        self.assertIsInstance(combined_fisher, pd.DataFrame)
        self.assertIn('gene_names', combined_fisher.columns)
        self.assertIn('number_samples', combined_fisher.columns)
        self.assertIn('calibrated_p_values', combined_fisher.columns)
        
        # All genes should appear in 3 samples since we used the same gene set
        self.assertTrue(all(combined_fisher['number_samples'] == 3))
        
        # Check that we have the expected number of genes
        expected_n_genes = input_exp_mat_raw.shape[1]
        self.assertEqual(len(combined_fisher), expected_n_genes)
        
        # Combine p-values using Stouffer's method
        print("Combining p-values using Stouffer's method...")
        combined_stouffer = combine_p_values(
            [p_values_sample1, p_values_sample2, p_values_sample3],
            method="stouffer"
        )
        
        # Validate Stouffer results
        self.assertIsInstance(combined_stouffer, pd.DataFrame)
        self.assertEqual(len(combined_stouffer), expected_n_genes)
        
        # Check that all p-values are valid (between 0 and 1)
        self.assertTrue(all(0 <= p <= 1 for p in combined_fisher['calibrated_p_values']))
        self.assertTrue(all(0 <= p <= 1 for p in combined_stouffer['calibrated_p_values']))
        
        # Test with subsets of samples (simulating missing data)
        # Take only first half of genes from sample 2
        half_genes = len(p_values_sample2) // 2
        p_values_sample2_subset = p_values_sample2.iloc[:half_genes]
        
        combined_missing = combine_p_values(
            [p_values_sample1, p_values_sample2_subset, p_values_sample3],
            method="fisher"
        )
        
        # Check that genes have different number_samples values
        gene_counts = combined_missing['number_samples'].value_counts()
        self.assertIn(3, gene_counts.index)  # Some genes in all 3 samples
        self.assertIn(2, gene_counts.index)  # Some genes in only 2 samples
        
        print("Integration test completed successfully!")


class TestCombinePValues(unittest.TestCase):
    def test_fisher_method_basic(self):
        df1 = pd.DataFrame({
            'gene_names': ['A', 'B', 'C'],
            'p_values': [0.01, 0.20, 0.03]
        })
        df2 = pd.DataFrame({
            'gene_names': ['A', 'C', 'D'],
            'p_values': [0.04, 0.10, 0.50]
        })
        df3 = pd.DataFrame({
            'gene_names': ['B', 'C', 'E'],
            'p_values': [0.05, 0.02, 0.80]
        })
        
        result = combine_p_values([df1, df2, df3], method="fisher")
        
        # Check structure
        self.assertIsInstance(result, pd.DataFrame)
        self.assertListEqual(
            list(result.columns), 
            ['gene_names', 'number_samples', 'calibrated_p_values']
        )
        
        # Check gene names
        expected_genes = {'A', 'B', 'C', 'D', 'E'}
        self.assertEqual(set(result['gene_names']), expected_genes)
        
        # Check number of samples
        gene_samples = dict(zip(result['gene_names'], result['number_samples']))
        self.assertEqual(gene_samples['A'], 2)  # A appears in df1 and df2
        self.assertEqual(gene_samples['B'], 2)  # B appears in df1 and df3
        self.assertEqual(gene_samples['C'], 3)  # C appears in all three
        self.assertEqual(gene_samples['D'], 1)  # D appears only in df2
        self.assertEqual(gene_samples['E'], 1)  # E appears only in df3
        
        # Check that combined p-values are calculated
        self.assertTrue(all(pd.notna(result['calibrated_p_values'])))
        
        # Check that all p-values are between 0 and 1
        self.assertTrue(all(0 <= p <= 1 for p in result['calibrated_p_values']))
    
    def test_stouffer_method_basic(self):
        df1 = pd.DataFrame({
            'gene_names': ['A', 'B', 'C'],
            'p_values': [0.01, 0.20, 0.03]
        })
        df2 = pd.DataFrame({
            'gene_names': ['A', 'C', 'D'],
            'p_values': [0.04, 0.10, 0.50]
        })
        
        result = combine_p_values([df1, df2], method="stouffer")
        
        # Check structure
        self.assertIsInstance(result, pd.DataFrame)
        self.assertListEqual(
            list(result.columns), 
            ['gene_names', 'number_samples', 'calibrated_p_values']
        )
        
        # Check that all p-values are between 0 and 1
        self.assertTrue(all(0 <= p <= 1 for p in result['calibrated_p_values']))
    
    def test_single_dataframe(self):
        df = pd.DataFrame({
            'gene_names': ['A', 'B', 'C'],
            'p_values': [0.01, 0.05, 0.10]
        })
        
        result = combine_p_values([df], method="fisher")
        
        # With single input, combined p-values should be same as original
        self.assertEqual(len(result), 3)
        self.assertTrue(all(result['number_samples'] == 1))
        
        # Check that p-values are preserved (approximately)
        for i, gene in enumerate(['A', 'B', 'C']):
            row = result[result['gene_names'] == gene]
            self.assertAlmostEqual(
                row['calibrated_p_values'].iloc[0], 
                df['p_values'].iloc[i],
                places=10
            )
    
    def test_empty_list(self):
        result = combine_p_values([], method="fisher")
        
        self.assertIsInstance(result, pd.DataFrame)
        self.assertEqual(len(result), 0)
        self.assertListEqual(
            list(result.columns), 
            ['gene_names', 'number_samples', 'calibrated_p_values']
        )
    
    def test_missing_genes(self):
        df1 = pd.DataFrame({
            'gene_names': ['A', 'B'],
            'p_values': [0.01, 0.02]
        })
        df2 = pd.DataFrame({
            'gene_names': ['B', 'C'],
            'p_values': [0.03, 0.04]
        })
        
        result = combine_p_values([df1, df2], method="fisher")
        
        # Should have all three genes
        self.assertEqual(len(result), 3)
        self.assertEqual(set(result['gene_names']), {'A', 'B', 'C'})
        
        # Check number of samples
        gene_samples = dict(zip(result['gene_names'], result['number_samples']))
        self.assertEqual(gene_samples['A'], 1)
        self.assertEqual(gene_samples['B'], 2)
        self.assertEqual(gene_samples['C'], 1)
    
    def test_invalid_method(self):
        df = pd.DataFrame({
            'gene_names': ['A'],
            'p_values': [0.05]
        })
        
        with self.assertRaises(ValueError) as context:
            combine_p_values([df], method="invalid")
        
        self.assertIn("Method must be 'fisher' or 'stouffer'", str(context.exception))
    
    def test_invalid_dataframe_structure(self):
        df1 = pd.DataFrame({
            'gene_names': ['A', 'B'],
            'p_values': [0.01, 0.02]
        })
        df2 = pd.DataFrame({
            'genes': ['B', 'C'],  # Wrong column name
            'pvals': [0.03, 0.04]  # Wrong column name
        })
        
        with self.assertRaises(ValueError) as context:
            combine_p_values([df1, df2], method="fisher")
        
        self.assertIn("must have 'gene_names' and 'p_values' columns", str(context.exception))
    
    def test_extreme_p_values(self):
        df1 = pd.DataFrame({
            'gene_names': ['A', 'B', 'C'],
            'p_values': [0.0, 0.5, 1.0]
        })
        df2 = pd.DataFrame({
            'gene_names': ['A', 'B', 'C'],
            'p_values': [0.0, 0.5, 1.0]
        })
        
        # Test both methods with extreme values
        for method in ["fisher", "stouffer"]:
            result = combine_p_values([df1, df2], method=method)
            
            # Check that no NaN or inf values are produced
            self.assertTrue(all(pd.notna(result['calibrated_p_values'])))
            self.assertTrue(all(np.isfinite(result['calibrated_p_values'])))
            
            # Check that all p-values are still between 0 and 1
            self.assertTrue(all(0 <= p <= 1 for p in result['calibrated_p_values']))
    
    def test_fisher_vs_stouffer_comparison(self):
        df1 = pd.DataFrame({
            'gene_names': ['A', 'B', 'C'],
            'p_values': [0.01, 0.05, 0.10]
        })
        df2 = pd.DataFrame({
            'gene_names': ['A', 'B', 'C'],
            'p_values': [0.02, 0.03, 0.15]
        })
        
        result_fisher = combine_p_values([df1, df2], method="fisher")
        result_stouffer = combine_p_values([df1, df2], method="stouffer")
        
        # Both should have same structure
        self.assertEqual(len(result_fisher), len(result_stouffer))
        self.assertEqual(set(result_fisher['gene_names']), set(result_stouffer['gene_names']))
        
        # Both should produce valid p-values
        self.assertTrue(all(0 <= p <= 1 for p in result_fisher['calibrated_p_values']))
        self.assertTrue(all(0 <= p <= 1 for p in result_stouffer['calibrated_p_values']))
        
        # For very small p-values, Fisher's method should give smaller combined p-values
        # This is a general trend, not always true
        gene_a_fisher = result_fisher[result_fisher['gene_names'] == 'A']['calibrated_p_values'].iloc[0]
        gene_a_stouffer = result_stouffer[result_stouffer['gene_names'] == 'A']['calibrated_p_values'].iloc[0]
        
        # Both should be small since input p-values are small
        self.assertLess(gene_a_fisher, 0.05)
        self.assertLess(gene_a_stouffer, 0.05)


if __name__ == "__main__":
    unittest.main()
