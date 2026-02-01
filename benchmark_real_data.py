"""
Benchmark script using real test data scaled up to larger sizes.
Compares original vs optimized scBSP implementations.
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


def _calculate_sparse_variances(input_csr_mat: csr_matrix, axis: int):
    input_csr_mat_squared = input_csr_mat.copy()
    input_csr_mat_squared.data **= 2
    return input_csr_mat_squared.mean(axis) - np.square(input_csr_mat.mean(axis))


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

    input_exp_mat_norm = _scale_sparse_matrix_original(input_exp_mat_raw).transpose()
    input_exp_mat_raw_t = input_exp_mat_raw.transpose()

    def _get_inverted_diag_matrix(sum_axis_0: np.ndarray) -> csr_matrix:
        with np.errstate(divide="ignore", invalid="ignore"):
            diag_data = np.reciprocal(sum_axis_0, where=sum_axis_0 != 0)
        return diags(diag_data, offsets=0, format="csr")

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


# ============================================================================
# DATA LOADING AND SCALING
# ============================================================================

def load_and_scale_data(csv_path: str, scale_factor: int = 1):
    """
    Load test data and optionally scale it up by duplicating rows.
    
    Args:
        csv_path: Path to CSV file
        scale_factor: How many times to duplicate the data
    
    Returns:
        input_sp_mat: Spatial coordinates (N x 3)
        input_exp_mat: Expression matrix as DataFrame
    """
    print(f"Loading data from {csv_path}...")
    data = pd.read_csv(csv_path)
    
    original_cells = len(data)
    original_genes = len(data.columns) - 3  # Subtract x, y, z columns
    
    if scale_factor > 1:
        # Duplicate rows with small random offsets to spatial coordinates
        dfs = [data]
        np.random.seed(42)
        for i in range(1, scale_factor):
            df_copy = data.copy()
            # Add small random offsets to coordinates to avoid identical points
            df_copy['x'] = df_copy['x'] + np.random.uniform(-0.1, 0.1, len(df_copy))
            df_copy['y'] = df_copy['y'] + np.random.uniform(-0.1, 0.1, len(df_copy))
            df_copy['z'] = df_copy['z'] + np.random.uniform(-0.1, 0.1, len(df_copy))
            dfs.append(df_copy)
        data = pd.concat(dfs, ignore_index=True)
    
    input_sp_mat = data[["x", "y", "z"]].to_numpy()
    input_exp_mat = data.iloc[:, 3:]
    
    print(f"  Original: {original_cells} cells, {original_genes} genes")
    print(f"  Scaled:   {len(data)} cells, {len(input_exp_mat.columns)} genes (scale={scale_factor}x)")
    
    return input_sp_mat, input_exp_mat


# ============================================================================
# BENCHMARKING
# ============================================================================

def benchmark_with_real_data(csv_path: str, scale_factor: int, n_runs: int = 2):
    """Benchmark using real test data."""
    from scbsp import granp as granp_optimized
    
    print(f"\n{'='*70}")
    print(f"Benchmark: Real data with {scale_factor}x scaling")
    print(f"{'='*70}")
    
    # Load and scale data
    input_sp_mat, input_exp_mat = load_and_scale_data(csv_path, scale_factor)
    n_cells = len(input_sp_mat)
    n_genes = len(input_exp_mat.columns)
    
    # Warm-up
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
    
    print(f"\n{'─'*50}")
    print(f"Original avg:  {original_avg:.3f}s")
    print(f"Optimized avg: {optimized_avg:.3f}s")
    print(f"Speedup:       {speedup:.2f}x")
    
    # Verify correctness
    print(f"\n{'─'*50}")
    print("Verifying correctness...")
    
    result_original = result_original.sort_values('gene_names').reset_index(drop=True)
    result_optimized = result_optimized.sort_values('gene_names').reset_index(drop=True)
    
    gene_match = (result_original['gene_names'] == result_optimized['gene_names']).all()
    p_value_diff = np.abs(result_original['p_values'] - result_optimized['p_values'])
    max_diff = p_value_diff.max()
    mean_diff = p_value_diff.mean()
    
    print(f"Gene names match: {gene_match}")
    print(f"P-value max difference: {max_diff:.2e}")
    print(f"P-value mean difference: {mean_diff:.2e}")
    
    # Check if SVG detection is consistent
    threshold = 0.0001
    original_svg = set(result_original[result_original['p_values'] < threshold]['gene_names'])
    optimized_svg = set(result_optimized[result_optimized['p_values'] < threshold]['gene_names'])
    svg_match = original_svg == optimized_svg
    
    print(f"SVG genes (p<{threshold}): {len(original_svg)} original, {len(optimized_svg)} optimized")
    print(f"SVG sets match: {svg_match}")
    
    if max_diff < 1e-10:
        print("✅ Results are numerically identical")
    elif max_diff < 1e-6:
        print("✅ Results match within floating-point tolerance")
    else:
        print("⚠️  Results have some differences")
    
    return {
        'n_cells': n_cells,
        'n_genes': n_genes,
        'scale_factor': scale_factor,
        'original_time': original_avg,
        'optimized_time': optimized_avg,
        'speedup': speedup,
        'max_diff': max_diff,
        'correct': max_diff < 1e-6,
        'svg_match': svg_match
    }


if __name__ == "__main__":
    csv_path = "test/test_data/scenario1_RW1_3-5_1.csv"
    
    print("=" * 70)
    print("scBSP Performance Benchmark: Real Data at Multiple Scales")
    print("=" * 70)
    
    results = []
    
    # Original size (1x)
    results.append(benchmark_with_real_data(csv_path, scale_factor=1, n_runs=3))
    
    # 2x size
    results.append(benchmark_with_real_data(csv_path, scale_factor=2, n_runs=2))
    
    # 4x size  
    results.append(benchmark_with_real_data(csv_path, scale_factor=4, n_runs=2))
    
    # 8x size
    results.append(benchmark_with_real_data(csv_path, scale_factor=8, n_runs=2))
    
    # Summary
    print("\n" + "=" * 70)
    print("SUMMARY - Real Data Benchmark Results")
    print("=" * 70)
    print(f"\n{'Scale':<8} {'Cells':<10} {'Genes':<8} {'Original':<12} {'Optimized':<12} {'Speedup':<10} {'Correct'}")
    print("-" * 70)
    for r in results:
        print(f"{r['scale_factor']}x{'':<6} {r['n_cells']:<10} {r['n_genes']:<8} "
              f"{r['original_time']:.3f}s{'':<7} {r['optimized_time']:.3f}s{'':<7} "
              f"{r['speedup']:.2f}x{'':<6} {'✅' if r['correct'] else '❌'}")
