"""
CPU vs GPU Benchmark for scBSP

This script benchmarks the granp function comparing CPU and GPU performance
across different data sizes.

Usage:
    python benchmark_cpu_gpu.py

Requirements:
    - PyTorch with CUDA support
    - scbsp package installed (pip install -e .)
"""

import time
import sys
import numpy as np
import pandas as pd
from scipy.sparse import random as sparse_random
from typing import Tuple, Dict, List, Optional
import warnings

warnings.filterwarnings('ignore')

CUDA_AVAILABLE = False
GPU_NAME = "N/A"
GPU_MEMORY = 0
GPU_BACKEND = "none"

try:
    import cupy as cp
    if cp.cuda.is_available():
        CUDA_AVAILABLE = True
        GPU_BACKEND = "cupy"
        GPU_NAME = cp.cuda.runtime.getDeviceProperties(0)['name'].decode()
        GPU_MEMORY = cp.cuda.runtime.getDeviceProperties(0)['totalGlobalMem'] / (1024**3)
except ImportError:
    pass

if not CUDA_AVAILABLE:
    try:
        import torch
        if torch.cuda.is_available():
            CUDA_AVAILABLE = True
            GPU_BACKEND = "torch"
            GPU_NAME = torch.cuda.get_device_name(0)
            GPU_MEMORY = torch.cuda.get_device_properties(0).total_memory / (1024**3)
    except ImportError:
        pass

# Import scBSP
import sys
import os

parent_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if parent_dir not in sys.path:
    sys.path.insert(0, parent_dir)

from scbsp.scbsp import granp, gpu_backend as scbsp_gpu_backend


def generate_synthetic_data(
    n_cells: int, 
    n_genes: int, 
    sparsity: float = 0.8,
    n_dims: int = 3,
    seed: int = 42
) -> Tuple[np.ndarray, pd.DataFrame]:
    """
    Generate synthetic spatial transcriptomics data.
    
    Args:
        n_cells: Number of cells/spots
        n_genes: Number of genes
        sparsity: Fraction of zeros in expression matrix (0-1)
        n_dims: Number of spatial dimensions (2 or 3)
        seed: Random seed for reproducibility
        
    Returns:
        Tuple of (spatial_coordinates, expression_matrix)
    """
    np.random.seed(seed)
    
    # Generate spatial coordinates (uniformly distributed)
    spatial_coords = np.random.uniform(0, 100, size=(n_cells, n_dims))
    
    # Generate sparse expression matrix
    density = 1 - sparsity
    exp_sparse = sparse_random(n_cells, n_genes, density=density, format='csr', dtype=np.float32)
    
    # Scale to realistic expression values (0-1000)
    exp_sparse.data = exp_sparse.data * 1000
    
    # Convert to DataFrame with gene names
    gene_names = [f"Gene_{i}" for i in range(n_genes)]
    exp_df = pd.DataFrame(exp_sparse.toarray(), columns=gene_names)
    
    return spatial_coords, exp_df


def run_single_benchmark(
    n_cells: int,
    n_genes: int,
    use_gpu: bool,
    n_runs: int = 3
) -> Dict:
    """
    Run a single benchmark configuration.
    
    Args:
        n_cells: Number of cells
        n_genes: Number of genes
        use_gpu: Whether to use GPU
        n_runs: Number of runs for timing
        
    Returns:
        Dictionary with benchmark results
    """
    # Generate data
    spatial_coords, exp_df = generate_synthetic_data(n_cells, n_genes)
    
    if use_gpu and CUDA_AVAILABLE:
        try:
            _ = granp(spatial_coords, exp_df, use_gpu=True)
            if scbsp_gpu_backend == "cupy":
                import cupy as cp
                cp.cuda.Stream.null.synchronize()
            elif scbsp_gpu_backend and scbsp_gpu_backend.startswith("torch"):
                import torch
                torch.cuda.synchronize()
        except Exception as e:
            return {
                'n_cells': n_cells,
                'n_genes': n_genes,
                'use_gpu': use_gpu,
                'time_seconds': float('nan'),
                'error': str(e)
            }
    
    times = []
    for run in range(n_runs):
        if use_gpu and CUDA_AVAILABLE:
            if scbsp_gpu_backend == "cupy":
                import cupy as cp
                cp.get_default_memory_pool().free_all_blocks()
            elif scbsp_gpu_backend and scbsp_gpu_backend.startswith("torch"):
                import torch
                torch.cuda.empty_cache()
        
        start_time = time.perf_counter()
        
        try:
            result = granp(spatial_coords, exp_df, use_gpu=use_gpu)
            
            # Synchronize GPU if used
            if use_gpu and CUDA_AVAILABLE:
                if scbsp_gpu_backend == "cupy":
                    import cupy as cp
                    cp.cuda.Stream.null.synchronize()
                elif scbsp_gpu_backend and scbsp_gpu_backend.startswith("torch"):
                    import torch
                    torch.cuda.synchronize()
                
            end_time = time.perf_counter()
            times.append(end_time - start_time)
            
        except Exception as e:
            return {
                'n_cells': n_cells,
                'n_genes': n_genes,
                'use_gpu': use_gpu,
                'time_seconds': float('nan'),
                'error': str(e)
            }
    
    # Calculate statistics
    mean_time = np.mean(times)
    std_time = np.std(times)
    
    # Get GPU memory usage if applicable
    gpu_memory_used = 0
    if use_gpu and CUDA_AVAILABLE:
        if scbsp_gpu_backend == "cupy":
            import cupy as cp
            gpu_memory_used = cp.get_default_memory_pool().used_bytes() / (1024**3)  # GB
        elif scbsp_gpu_backend and scbsp_gpu_backend.startswith("torch"):
            import torch
            gpu_memory_used = torch.cuda.max_memory_allocated() / (1024**3)  # GB
            torch.cuda.reset_peak_memory_stats()
    
    return {
        'n_cells': n_cells,
        'n_genes': n_genes,
        'use_gpu': use_gpu,
        'time_seconds': mean_time,
        'std_seconds': std_time,
        'gpu_memory_gb': gpu_memory_used,
        'error': None
    }


def print_header():
    """Print benchmark header information."""
    print("=" * 70)
    print("scBSP CPU vs GPU Benchmark")
    print("=" * 70)
    print(f"\nSystem Information:")
    print(f"  CUDA Available: {CUDA_AVAILABLE}")
    print(f"  GPU Backend: {scbsp_gpu_backend}")
    if CUDA_AVAILABLE:
        print(f"  GPU: {GPU_NAME}")
        print(f"  GPU Memory: {GPU_MEMORY:.1f} GB")
    print(f"  NumPy version: {np.__version__}")
    try:
        import scipy
        print(f"  SciPy version: {scipy.__version__}")
    except:
        pass
    if scbsp_gpu_backend == "cupy":
        import cupy
        print(f"  CuPy version: {cupy.__version__}")
    elif scbsp_gpu_backend and scbsp_gpu_backend.startswith("torch"):
        import torch
        print(f"  PyTorch version: {torch.__version__}")
        print(f"  CUDA version: {torch.version.cuda}")
    print()


def format_time(seconds: float) -> str:
    """Format time in human-readable format."""
    if np.isnan(seconds):
        return "ERROR"
    if seconds < 60:
        return f"{seconds:.2f}s"
    else:
        return f"{seconds/60:.2f}m"


def run_benchmarks():
    """Run the full benchmark suite."""
    print_header()
    
    # Define test configurations
    # (n_cells, n_genes)
    configurations = [
        (500, 200),      # Tiny - quick sanity check
        (1000, 500),     # Small
        (2000, 1000),    # Medium-small
        (5000, 1000),    # Medium
        (10000, 1000),   # Medium-large
    ]
    
    # Add larger configs if GPU is available (they'd be too slow on CPU)
    if CUDA_AVAILABLE:
        configurations.extend([
            (20000, 1000),   # Large
            (30000, 1000),   # Extra large
        ])
    
    results: List[Dict] = []
    
    print("Running Benchmarks...")
    print("-" * 70)
    print(f"{'Config':<15} {'Device':<8} {'Time':>10} {'Std':>8} {'GPU Mem':>10} {'Status':<10}")
    print("-" * 70)
    
    for n_cells, n_genes in configurations:
        config_str = f"{n_cells}x{n_genes}"
        
        # CPU benchmark
        print(f"{config_str:<15} {'CPU':<8}", end="", flush=True)
        result_cpu = run_single_benchmark(n_cells, n_genes, use_gpu=False, n_runs=3)
        results.append(result_cpu)
        
        if result_cpu['error']:
            err_msg = result_cpu['error'][:20]
            print(f"{'ERROR':>10} {'':>8} {'':>10} {err_msg}")
        else:
            time_str = format_time(result_cpu['time_seconds'])
            std_str = '±' + f"{result_cpu['std_seconds']:.2f}"
            print(f"{time_str:>10} {std_str:>8} {'N/A':>10} {'OK':<10}")
        
        # GPU benchmark (if available)
        if CUDA_AVAILABLE:
            print(f"{'':<15} {'GPU':<8}", end="", flush=True)
            result_gpu = run_single_benchmark(n_cells, n_genes, use_gpu=True, n_runs=3)
            results.append(result_gpu)
            
            if result_gpu['error']:
                err_msg = result_gpu['error'][:20]
                print(f"{'ERROR':>10} {'':>8} {'':>10} {err_msg}")
            else:
                speedup = result_cpu['time_seconds'] / result_gpu['time_seconds'] if result_gpu['time_seconds'] > 0 else 0
                gpu_mem_str = f"{result_gpu['gpu_memory_gb']:.2f}GB" if result_gpu['gpu_memory_gb'] > 0 else "N/A"
                time_str = format_time(result_gpu['time_seconds'])
                std_str = '±' + f"{result_gpu['std_seconds']:.2f}"
                speedup_str = f"{speedup:.2f}x"
                print(f"{time_str:>10} {std_str:>8} {gpu_mem_str:>10} {speedup_str:<10}")
    
    print("-" * 70)
    
    # Summary statistics
    print("\n" + "=" * 70)
    print("Summary")
    print("=" * 70)
    
    if CUDA_AVAILABLE:
        print("\nSpeedup Analysis (CPU Time / GPU Time):")
        print("-" * 40)
        
        for i in range(0, len(results), 2):
            cpu_result = results[i]
            gpu_result = results[i + 1] if i + 1 < len(results) else None
            
            if gpu_result and not cpu_result['error'] and not gpu_result['error']:
                speedup = cpu_result['time_seconds'] / gpu_result['time_seconds']
                config = f"{cpu_result['n_cells']}x{cpu_result['n_genes']}"
                
                bar_len = int(min(speedup * 5, 30))
                bar = "█" * bar_len
                
                print(f"  {config:<15} {speedup:>6.2f}x  {bar}")
    else:
        print("\nGPU not available. To enable GPU benchmarks, install PyTorch with CUDA support:")
        print("  pip install torch --index-url https://download.pytorch.org/whl/cu121")
    
    # Save results to CSV
    results_df = pd.DataFrame(results)
    output_file = "benchmark_results.csv"
    results_df.to_csv(output_file, index=False)
    print(f"\nResults saved to: {output_file}")
    
    return results


if __name__ == "__main__":
    results = run_benchmarks()
