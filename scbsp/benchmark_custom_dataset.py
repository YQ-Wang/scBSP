import pandas as pd
import numpy as np
import time
import sys
import os
from scipy.sparse import csr_matrix

parent_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if parent_dir not in sys.path:
    sys.path.insert(0, parent_dir)

from scbsp.scbsp import granp, gpu_enabled, gpu_backend

def run_benchmark(dataset_path: str, duplication_factors: list):
    print("=" * 80)
    print(f"Benchmarking Dataset: {dataset_path}")
    print(f"GPU Backend: {gpu_backend}")
    print("=" * 80)
    
    if not os.path.exists(dataset_path):
        print(f"Error: Dataset {dataset_path} not found.")
        return

    # Load original data
    print("Loading original dataset...")
    df_orig = pd.read_csv(dataset_path)
    print(f"Original shape: {df_orig.shape}")
    
    if all(col in df_orig.columns for col in ['x', 'y', 'z']):
        coord_cols = ['x', 'y', 'z']
        expr_df = df_orig.drop(columns=coord_cols)
    else:
        coord_cols = df_orig.columns[:3].tolist()
        expr_df = df_orig.iloc[:, 3:]
        
    print(f"Coordinate columns: {coord_cols}")
    print(f"Gene expression columns: {expr_df.shape[1]}")

    results = []

    for factor in duplication_factors:
        print(f"\n--- Testing with duplication factor {factor}x ---")
        
        if factor == 1:
            df_current = df_orig
        else:
            df_current = pd.concat([df_orig] * factor, ignore_index=True)
            for col in coord_cols:
                noise = np.random.normal(0, 0.0001, size=len(df_current))
                df_current[col] = df_current[col] + noise
        
        current_sp = df_current[coord_cols].to_numpy()
        current_exp = df_current.drop(columns=coord_cols)
        
        n_cells = current_sp.shape[0]
        n_genes = current_exp.shape[1]
        print(f"Current shape: {n_cells} cells x {n_genes} genes")

        # Run CPU
        print("Running CPU...")
        start_cpu = time.perf_counter()
        res_cpu = granp(current_sp, current_exp, use_gpu=False)
        end_cpu = time.perf_counter()
        cpu_time = end_cpu - start_cpu
        print(f"CPU Time: {cpu_time:.2f}s")

        # Run GPU
        gpu_time = None
        correlation = None
        max_diff = None
        
        if gpu_enabled:
            print(f"Running GPU ({gpu_backend})...")
            try:
                # Warm up
                _ = granp(current_sp[:100] if n_cells > 100 else current_sp, 
                          current_exp.iloc[:100] if n_cells > 100 else current_exp, 
                          use_gpu=True)
                
                start_gpu = time.perf_counter()
                res_gpu = granp(current_sp, current_exp, use_gpu=True)
                end_gpu = time.perf_counter()
                gpu_time = end_gpu - start_gpu
                print(f"GPU Time: {gpu_time:.2f}s")
                
                cpu_pvals = res_cpu['p_values'].values
                gpu_pvals = res_gpu['p_values'].values
                
                correlation = np.corrcoef(cpu_pvals, gpu_pvals)[0, 1]
                max_diff = np.max(np.abs(cpu_pvals - gpu_pvals))
                
                print(f"Correlation (CPU vs GPU): {correlation:.6f}")
                print(f"Max Difference: {max_diff:.2e}")
                
            except Exception as e:
                print(f"GPU Error: {e}")
        else:
            print("GPU not enabled, skipping GPU run.")

        results.append({
            'factor': factor,
            'n_cells': n_cells,
            'n_genes': n_genes,
            'cpu_time': cpu_time,
            'gpu_time': gpu_time,
            'correlation': correlation,
            'max_diff': max_diff
        })

    # Summary table
    print("\n" + "=" * 80)
    print("BENCHMARK SUMMARY")
    print("=" * 80)
    print(f"{'Factor':<8} {'Cells':<10} {'CPU(s)':<10} {'GPU(s)':<10} {'Speedup':<10} {'Corr':<10}")
    for res in results:
        speedup = res['cpu_time'] / res['gpu_time'] if res['gpu_time'] else 0
        corr_str = f"{res['correlation']:.6f}" if res['correlation'] is not None else "N/A"
        gpu_str = f"{res['gpu_time']:.2f}" if res['gpu_time'] is not None else "N/A"
        print(f"{res['factor']:<8} {res['n_cells']:<10} {res['cpu_time']:<10.2f} {gpu_str:<10} {speedup:<10.2f} {corr_str:<10}")
    print("=" * 80)

if __name__ == "__main__":
    # Default to the test data if none specified
    dataset = "test/test_data/scenario1_RW1_3-5_1.csv"
    
    # If benchmark_results.csv was actually intended as the DATA source, 
    # we would use it, but it's likely a mistake.
    # However, I'll check if it exists in the root.
    target_dataset = "benchmark_results.csv"
    
    # Re-evaluating: the user specifically tagged @[benchmark_results.csv].
    # Let me check if benchmark_results.csv is actually a valid dataset (with coords and expr).
    try:
        check_df = pd.read_csv(target_dataset)
        if 'x' in check_df.columns or check_df.shape[1] > 100:
            dataset = target_dataset
            print(f"Using tagged file {target_dataset} as dataset.")
        else:
            print(f"Tagged file {target_dataset} doesn't look like a spatial dataset. Using {dataset} instead.")
    except:
        print(f"Could not read {target_dataset}. Using {dataset} instead.")

    run_benchmark(dataset, [1, 2, 3, 4])
