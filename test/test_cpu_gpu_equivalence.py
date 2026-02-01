"""
Test to verify CPU and GPU results are equivalent and benchmark larger datasets.
"""

import numpy as np
import pandas as pd
from scipy.sparse import random as sparse_random
import sys
import os

parent_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if parent_dir not in sys.path:
    sys.path.insert(0, parent_dir)

from scbsp.scbsp import granp, gpu_enabled, gpu_backend


def generate_test_data(n_cells: int, n_genes: int, sparsity: float = 0.8, seed: int = 42):
    """Generate synthetic spatial transcriptomics data."""
    np.random.seed(seed)
    
    spatial_coords = np.random.uniform(0, 100, size=(n_cells, 3))
    
    density = 1 - sparsity
    exp_sparse = sparse_random(n_cells, n_genes, density=density, format='csr', dtype=np.float32)
    exp_sparse.data = exp_sparse.data * 1000
    
    gene_names = [f"Gene_{i}" for i in range(n_genes)]
    exp_df = pd.DataFrame(exp_sparse.toarray(), columns=gene_names)
    
    return spatial_coords, exp_df


def test_cpu_gpu_equivalence():
    """Test that CPU and GPU produce equivalent results."""
    print("=" * 70)
    print("Testing CPU/GPU Result Equivalence")
    print("=" * 70)
    print(f"\nGPU Backend: {gpu_backend}")
    print(f"GPU Enabled: {gpu_enabled}")
    
    if not gpu_enabled:
        print("\nGPU not available, skipping GPU tests.")
        return True
    
    test_configs = [
        (500, 200),
        (1000, 500),
        (2000, 500),
        (5000, 500),
    ]
    
    all_passed = True
    
    for n_cells, n_genes in test_configs:
        print(f"\nTesting {n_cells} cells x {n_genes} genes...")
        
        spatial_coords, exp_df = generate_test_data(n_cells, n_genes)
        
        # Run CPU
        try:
            result_cpu = granp(spatial_coords, exp_df, use_gpu=False)
        except Exception as e:
            print(f"  CPU ERROR: {e}")
            all_passed = False
            continue
        
        # Run GPU
        try:
            result_gpu = granp(spatial_coords, exp_df, use_gpu=True)
        except Exception as e:
            print(f"  GPU ERROR: {e}")
            all_passed = False
            continue
        
        cpu_pvals = result_cpu['p_values'].values
        gpu_pvals = result_gpu['p_values'].values
        
        # Check if gene names match
        if not np.array_equal(result_cpu['gene_names'].values, result_gpu['gene_names'].values):
            print(f"  FAIL: Gene names don't match")
            all_passed = False
            continue
        
        max_diff = np.max(np.abs(cpu_pvals - gpu_pvals))
        mean_diff = np.mean(np.abs(cpu_pvals - gpu_pvals))
        correlation = np.corrcoef(cpu_pvals, gpu_pvals)[0, 1]
        
        tolerance = 1e-5
        
        if max_diff < tolerance or correlation > 0.9999:
            print(f"  PASS: Max diff = {max_diff:.2e}, Mean diff = {mean_diff:.2e}, Correlation = {correlation:.6f}")
        else:
            print(f"  FAIL: Max diff = {max_diff:.2e}, Mean diff = {mean_diff:.2e}, Correlation = {correlation:.6f}")
            all_passed = False
    
    print("\n" + "=" * 70)
    if all_passed:
        print("All CPU/GPU equivalence tests PASSED!")
    else:
        print("Some tests FAILED!")
    print("=" * 70)
    
    return all_passed


def test_large_dataset_optimization():
    """Test and optimize for larger datasets."""
    print("\n" + "=" * 70)
    print("Testing Large Dataset Handling")
    print("=" * 70)
    
    if not gpu_enabled:
        print("\nGPU not available, skipping large dataset tests.")
        return True
    
    import time
    
    # Test progressively larger datasets
    test_sizes = [10000, 15000, 20000, 25000, 30000]
    n_genes = 500  # Keep genes smaller to focus on cell scaling
    
    results = []
    
    for n_cells in test_sizes:
        print(f"\nTesting {n_cells} cells x {n_genes} genes...")
        
        spatial_coords, exp_df = generate_test_data(n_cells, n_genes)
        
        # Test CPU
        start = time.perf_counter()
        try:
            result_cpu = granp(spatial_coords, exp_df, use_gpu=False)
            cpu_time = time.perf_counter() - start
            print(f"  CPU: {cpu_time:.2f}s")
        except Exception as e:
            print(f"  CPU ERROR: {e}")
            cpu_time = None
        
        # Test GPU
        start = time.perf_counter()
        try:
            result_gpu = granp(spatial_coords, exp_df, use_gpu=True)
            gpu_time = time.perf_counter() - start
            print(f"  GPU: {gpu_time:.2f}s")
            
            # Verify equivalence
            if cpu_time is not None:
                correlation = np.corrcoef(
                    result_cpu['p_values'].values, 
                    result_gpu['p_values'].values
                )[0, 1]
                print(f"  Correlation: {correlation:.6f}")
        except Exception as e:
            print(f"  GPU ERROR: {str(e)[:50]}...")
            gpu_time = None
        
        results.append({
            'n_cells': n_cells,
            'cpu_time': cpu_time,
            'gpu_time': gpu_time
        })
    
    return results


    # Run large dataset test
if __name__ == "__main__":
    equiv_passed = test_cpu_gpu_equivalence()
    large_results = test_large_dataset_optimization()
    
    print("\n" + "=" * 70)
    print("Summary")
    print("=" * 70)
    print(f"Equivalence tests: {'PASSED' if equiv_passed else 'FAILED'}")
    
    if isinstance(large_results, list):
        success_count = len([r for r in large_results if r['gpu_time'] is not None])
        print(f"Large dataset tests completed: {success_count}/{len(large_results)} successful")
    else:
        print("Large dataset tests: SKIPPED (GPU not available)")
