"""
Benchmark script comparing original vs optimized scBSP implementations.
Tests both correctness (matching outputs) and performance (timing).
"""

import time
import numpy as np
import pandas as pd
from scipy.sparse import csr_matrix, diags, identity
from scipy.stats import gmean, lognorm, chi2, norm
from sklearn.neighbors import BallTree


# ============================================================================
# ORIGINAL IMPLEMENTATIONS (for comparison)
# ============================================================================

def _scale_sparse_matrix_original(input_exp_mat: csr_matrix) -> csr_matrix:
    """Original implementation with list comprehension."""
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

    data_scaled = data / np.repeat(row_max, row_indices)
    scaled_matrix = csr_matrix((data_scaled, (rows, cols)), shape=input_exp_mat.shape)
    return scaled_matrix


def _binary_distance_matrix_threshold_original(
    input_sparse_mat_array: np.ndarray, d_val: float, leaf_size: int
) -> csr_matrix:
    """Original implementation with generator."""
    ball_tree = BallTree(input_sparse_mat_array, leaf_size=leaf_size)
    indices = ball_tree.query_radius(
        input_sparse_mat_array, r=d_val, return_distance=False
    )
    
    def generate_data():
        for i, idx in enumerate(indices):
            yield from ((i, j, 1) for j in idx)

    rows, cols, data = zip(*generate_data())
    
    sparse_mat = csr_matrix(
        (data, (rows, cols)),
        shape=(input_sparse_mat_array.shape[0], input_sparse_mat_array.shape[0]),
        dtype=np.int8
    )

    return sparse_mat + identity(
        input_sparse_mat_array.shape[0], format="csr", dtype=np.int8
    )


def granp_original(
    input_sp_mat: np.ndarray,
    input_exp_mat_raw,
    d1: float = 1.0,
    d2: float = 3.0,
    leaf_size: int = 80,
) -> pd.DataFrame:
    """Original granp implementation for comparison."""
    
    if isinstance(input_exp_mat_raw, pd.DataFrame):
        gene_names = input_exp_mat_raw.columns.astype(str).tolist()
        input_exp_mat_raw = csr_matrix(input_exp_mat_raw)
    else:
        gene_names = [f"Gene_{i}" for i in range(input_exp_mat_raw.shape[1])]
        input_exp_mat_raw = csr_matrix(input_exp_mat_raw)

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

    # Use original implementations
    input_exp_mat_norm = _scale_sparse_matrix_original(input_exp_mat_raw).transpose()
    input_exp_mat_raw_t = input_exp_mat_raw.transpose()

    def _get_inverted_diag_matrix(sum_axis_0: np.ndarray) -> csr_matrix:
        with np.errstate(divide="ignore", invalid="ignore"):
            diag_data = np.reciprocal(sum_axis_0, where=sum_axis_0 != 0)
        return diags(diag_data, offsets=0, format="csr")

    def _calculate_sparse_variances(input_csr_mat: csr_matrix, axis: int):
        input_csr_mat_squared = input_csr_mat.copy()
        input_csr_mat_squared.data **= 2
        return input_csr_mat_squared.mean(axis) - np.square(input_csr_mat.mean(axis))

    def _var_local_means(d_val: float):
        patches_cells = _binary_distance_matrix_threshold_original(
            input_sp_mat, d_val, leaf_size
        )
        patches_cells_centroid = diags(
            (patches_cells.sum(axis=1) > 1).astype(np.float32).A.ravel(),
            offsets=0,
            format="csr",
        )
        patches_cells -= patches_cells_centroid
        sum_axis_0 = patches_cells.sum(axis=0).A.ravel()
        diag_matrix_sparse = _get_inverted_diag_matrix(sum_axis_0)
        x_kj = input_exp_mat_norm @ (patches_cells @ diag_matrix_sparse)
        del patches_cells, patches_cells_centroid, diag_matrix_sparse
        return _calculate_sparse_variances(x_kj, axis=1)

    var_x = np.column_stack([
        _var_local_means(d_val).A.ravel() for d_val in (d1, d2)
    ])
    var_x_0_add = _calculate_sparse_variances(input_exp_mat_raw_t, axis=1).A.ravel()
    var_x_0_add /= max(var_x_0_add)
    t_matrix_sum = ((var_x[:, 1] / var_x[:, 0]) * var_x_0_add).tolist()

    # Original p-value calculation
    t_matrix_sum_upper90 = np.quantile(t_matrix_sum, 0.90)
    t_matrix_sum_mid = (val for val in t_matrix_sum if val < t_matrix_sum_upper90)
    log_t_matrix_sum_mid = np.fromiter((np.log(val) for val in t_matrix_sum_mid), dtype=float)
    log_norm_params = (log_t_matrix_sum_mid.mean(), log_t_matrix_sum_mid.std(ddof=1))

    def p_value_generator():
        for val in t_matrix_sum:
            yield 1 - lognorm.cdf(val, scale=np.exp(log_norm_params[0]), s=log_norm_params[1])

    p_values = list(p_value_generator())

    return pd.DataFrame({"gene_names": gene_names, "p_values": p_values})


def combine_p_values_original(list_of_pvalues, method="fisher"):
    """Original combine_p_values with iterrows."""
    if method not in ["fisher", "stouffer"]:
        raise ValueError(f"Method must be 'fisher' or 'stouffer', got '{method}'")
    
    for i, df in enumerate(list_of_pvalues):
        if not isinstance(df, pd.DataFrame):
            raise ValueError(f"Element {i} in list_of_pvalues is not a DataFrame")
        if 'gene_names' not in df.columns or 'p_values' not in df.columns:
            raise ValueError(f"DataFrame {i} must have 'gene_names' and 'p_values' columns")
    
    if not list_of_pvalues:
        return pd.DataFrame(columns=['gene_names', 'number_samples', 'calibrated_p_values'])
    
    dfs_renamed = []
    for i, df in enumerate(list_of_pvalues):
        df_copy = df.copy()
        df_copy = df_copy.rename(columns={'p_values': f'p_values_{i+1}'})
        dfs_renamed.append(df_copy)
    
    merged = dfs_renamed[0]
    for df in dfs_renamed[1:]:
        merged = pd.merge(merged, df, on='gene_names', how='outer')
    
    pval_cols = [col for col in merged.columns if col.startswith('p_values_')]
    
    combined_results = []
    for _, row in merged.iterrows():
        gene_name = row['gene_names']
        pvals = [row[col] for col in pval_cols]
        valid_pvals = [p for p in pvals if pd.notna(p)]
        k = len(valid_pvals)
        
        if k == 0:
            combined_results.append({
                'gene_names': gene_name,
                'number_samples': 0,
                'calibrated_p_values': np.nan
            })
            continue
        
        if method == "fisher":
            epsilon = 1e-300
            valid_pvals_safe = [max(p, epsilon) for p in valid_pvals]
            stat = -2 * sum(np.log(valid_pvals_safe))
            combined_pval = 1 - chi2.cdf(stat, 2 * k)
        elif method == "stouffer":
            z_scores = []
            for p in valid_pvals:
                p_safe = max(min(p, 1 - 1e-15), 1e-15)
                z = norm.ppf(1 - p_safe/2) * np.sign(0.5 - p_safe)
                z_scores.append(z)
            z_combined = sum(z_scores) / np.sqrt(k)
            combined_pval = 2 * (1 - norm.cdf(abs(z_combined)))
        
        combined_results.append({
            'gene_names': gene_name,
            'number_samples': k,
            'calibrated_p_values': combined_pval
        })
    
    result_df = pd.DataFrame(combined_results)
    result_df['number_samples'] = result_df['number_samples'].astype(int)
    return result_df


# ============================================================================
# BENCHMARKING
# ============================================================================

def benchmark_granp(n_cells, n_genes, n_runs=3):
    """Benchmark granp function."""
    from scbsp import granp as granp_optimized
    
    print(f"\n{'='*60}")
    print(f"Benchmark: granp() with {n_cells} cells, {n_genes} genes")
    print(f"{'='*60}")
    
    # Generate test data
    np.random.seed(42)
    input_sp_mat = np.random.rand(n_cells, 3) * 100
    # Create sparse expression data (10% density)
    exp_data = np.random.rand(n_cells, n_genes)
    exp_data[exp_data < 0.9] = 0  # Make it sparse
    input_exp_mat = pd.DataFrame(exp_data, columns=[f"Gene_{i}" for i in range(n_genes)])
    
    # Warm-up run
    print("Warming up...")
    _ = granp_optimized(input_sp_mat[:100], input_exp_mat.iloc[:100, :50])
    
    # Benchmark original
    print(f"Running original implementation ({n_runs} runs)...")
    original_times = []
    for i in range(n_runs):
        start = time.time()
        result_original = granp_original(input_sp_mat, input_exp_mat)
        elapsed = time.time() - start
        original_times.append(elapsed)
        print(f"  Run {i+1}: {elapsed:.3f}s")
    
    # Benchmark optimized
    print(f"Running optimized implementation ({n_runs} runs)...")
    optimized_times = []
    for i in range(n_runs):
        start = time.time()
        result_optimized = granp_optimized(input_sp_mat, input_exp_mat)
        elapsed = time.time() - start
        optimized_times.append(elapsed)
        print(f"  Run {i+1}: {elapsed:.3f}s")
    
    # Calculate statistics
    original_avg = np.mean(original_times)
    optimized_avg = np.mean(optimized_times)
    speedup = original_avg / optimized_avg
    
    print(f"\n{'─'*40}")
    print(f"Original avg:  {original_avg:.3f}s")
    print(f"Optimized avg: {optimized_avg:.3f}s")
    print(f"Speedup:       {speedup:.2f}x")
    
    # Verify correctness
    print(f"\n{'─'*40}")
    print("Verifying correctness...")
    
    # Sort by gene name for comparison
    result_original = result_original.sort_values('gene_names').reset_index(drop=True)
    result_optimized = result_optimized.sort_values('gene_names').reset_index(drop=True)
    
    gene_match = (result_original['gene_names'] == result_optimized['gene_names']).all()
    p_value_diff = np.abs(result_original['p_values'] - result_optimized['p_values'])
    max_diff = p_value_diff.max()
    mean_diff = p_value_diff.mean()
    
    print(f"Gene names match: {gene_match}")
    print(f"P-value max difference: {max_diff:.2e}")
    print(f"P-value mean difference: {mean_diff:.2e}")
    
    if max_diff < 1e-10:
        print("✅ Results are numerically identical")
    elif max_diff < 1e-6:
        print("✅ Results match within floating-point tolerance")
    else:
        print("⚠️  Results have some differences")
    
    return {
        'n_cells': n_cells,
        'n_genes': n_genes,
        'original_time': original_avg,
        'optimized_time': optimized_avg,
        'speedup': speedup,
        'max_diff': max_diff,
        'correct': max_diff < 1e-6
    }


def benchmark_combine_p_values(n_genes, n_samples, n_runs=3):
    """Benchmark combine_p_values function."""
    from scbsp import combine_p_values as combine_optimized
    
    print(f"\n{'='*60}")
    print(f"Benchmark: combine_p_values() with {n_genes} genes, {n_samples} samples")
    print(f"{'='*60}")
    
    # Generate test data
    np.random.seed(42)
    dfs = []
    for i in range(n_samples):
        df = pd.DataFrame({
            'gene_names': [f'Gene_{j}' for j in range(n_genes)],
            'p_values': np.random.uniform(0, 1, n_genes)
        })
        dfs.append(df)
    
    # Benchmark original
    print(f"Running original implementation ({n_runs} runs)...")
    original_times = []
    for i in range(n_runs):
        start = time.time()
        result_original = combine_p_values_original(dfs, method="fisher")
        elapsed = time.time() - start
        original_times.append(elapsed)
        print(f"  Run {i+1}: {elapsed:.3f}s")
    
    # Benchmark optimized
    print(f"Running optimized implementation ({n_runs} runs)...")
    optimized_times = []
    for i in range(n_runs):
        start = time.time()
        result_optimized = combine_optimized(dfs, method="fisher")
        elapsed = time.time() - start
        optimized_times.append(elapsed)
        print(f"  Run {i+1}: {elapsed:.3f}s")
    
    # Calculate statistics
    original_avg = np.mean(original_times)
    optimized_avg = np.mean(optimized_times)
    speedup = original_avg / optimized_avg
    
    print(f"\n{'─'*40}")
    print(f"Original avg:  {original_avg:.3f}s")
    print(f"Optimized avg: {optimized_avg:.3f}s")
    print(f"Speedup:       {speedup:.2f}x")
    
    # Verify correctness
    print(f"\n{'─'*40}")
    print("Verifying correctness...")
    
    result_original = result_original.sort_values('gene_names').reset_index(drop=True)
    result_optimized = result_optimized.sort_values('gene_names').reset_index(drop=True)
    
    gene_match = (result_original['gene_names'] == result_optimized['gene_names']).all()
    p_value_diff = np.abs(result_original['calibrated_p_values'] - result_optimized['calibrated_p_values'])
    max_diff = p_value_diff.max()
    
    print(f"Gene names match: {gene_match}")
    print(f"P-value max difference: {max_diff:.2e}")
    
    if max_diff < 1e-10:
        print("✅ Results are numerically identical")
    elif max_diff < 1e-6:
        print("✅ Results match within floating-point tolerance")
    else:
        print("⚠️  Results have some differences")
    
    return {
        'n_genes': n_genes,
        'n_samples': n_samples,
        'original_time': original_avg,
        'optimized_time': optimized_avg,
        'speedup': speedup,
        'max_diff': max_diff,
        'correct': max_diff < 1e-6
    }


if __name__ == "__main__":
    print("=" * 60)
    print("scBSP Performance Benchmark: Original vs Optimized")
    print("=" * 60)
    
    # Run benchmarks with different sizes
    granp_results = []
    
    # Small dataset
    granp_results.append(benchmark_granp(n_cells=1000, n_genes=500, n_runs=3))
    
    # Medium dataset
    granp_results.append(benchmark_granp(n_cells=3000, n_genes=1000, n_runs=2))
    
    # Combine p-values benchmarks
    combine_results = []
    combine_results.append(benchmark_combine_p_values(n_genes=5000, n_samples=5, n_runs=3))
    combine_results.append(benchmark_combine_p_values(n_genes=10000, n_samples=10, n_runs=3))
    
    # Summary
    print("\n" + "=" * 60)
    print("SUMMARY")
    print("=" * 60)
    
    print("\ngranp() Results:")
    print("-" * 50)
    for r in granp_results:
        print(f"  {r['n_cells']:5d} cells, {r['n_genes']:4d} genes: "
              f"{r['speedup']:.2f}x speedup, correct={r['correct']}")
    
    print("\ncombine_p_values() Results:")
    print("-" * 50)
    for r in combine_results:
        print(f"  {r['n_genes']:5d} genes, {r['n_samples']:2d} samples: "
              f"{r['speedup']:.2f}x speedup, correct={r['correct']}")
