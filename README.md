# scBSP - A Fast Tool for Single-Cell Spatially Variable Genes Identifications on Large-Scale Spatially Resolved Transcriptomics Data

[![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.11123268.svg)](https://doi.org/10.5281/zenodo.11123268)

This package utilizes a granularity-based dimension-agnostic tool, single-cell big-small patch (scBSP), implementing sparse matrix operation and KD-tree/balltree method for distance calculation, for the identification of spatially variable genes on large-scale data.

## Installation

### Dependencies
To ensure scBSP functions optimally, the following dependencies are required:
- Python (>= 3.8)
- NumPy (>= 1.24.4)
- Pandas (>= 1.3.5)
- SciPy (>= 1.10.1)
- scikit-learn (>=1.3.2)

### Installation Commands
For Standard Installation (Using Ball Tree):

`pip install "scbsp"`

For Installation with GPU:

`pip install "scbsp[gpu]"`

## Usage

### Basic Usage

To use scBSP, you need to provide two primary inputs:

1. **Cell Coordinates Matrix (`input_sp_mat`)**: 
   - Format: Numpy array.
   - Dimensions: N x D, where N is the number of cells and D is the dimension of coordinates.

2. **Gene Expression Matrix (`input_exp_mat_raw`)**:
   - Format: Numpy array, Pandas DataFrame, or CSR matrix.
   - Dimensions: N x P, where N is the number of cells and P is the number of genes.

Additional parameters to specify include:

- `d1`: A floating-point number. Default value is 1.0.
- `d2`: A floating-point number. Default value is 3.0.
- `leaf_size`: Optional integer defining the maximum point threshold for the Ball Tree algorithm to revert to brute-force search (default = 80).
- `use_gpu`: Optional boolean defining whether to use the GPU (default = False).


### Example

Below is a straightforward example showcasing how to compute p-values with scBSP:

```python
import scbsp

# Load your data into these variables
input_sp_mat = ...  # Cell Coordinates Matrix
input_exp_mat_raw = ...  # Gene Expression Matrix

# Set the optional parameters
d1 = 1.0
d2 = 3.0

# Compute p-values
p_values = scbsp.granp(input_sp_mat, input_exp_mat_raw, d1, d2)
```

### Combining P-values Across Multiple Samples

When you have multiple samples or datasets and want to combine their p-values to identify consistently significant genes, you can use the `combine_p_values` function:

```python
import scbsp
import pandas as pd

# Assume you have p-values from three different samples
sample1_pvalues = scbsp.granp(sp_mat1, exp_mat1)
sample2_pvalues = scbsp.granp(sp_mat2, exp_mat2)
sample3_pvalues = scbsp.granp(sp_mat3, exp_mat3)

# Combine p-values using Fisher's method (default)
combined_results = scbsp.combine_p_values(
    [sample1_pvalues, sample2_pvalues, sample3_pvalues],
    method="fisher"
)

# Or use Stouffer's method
combined_results_stouffer = scbsp.combine_p_values(
    [sample1_pvalues, sample2_pvalues, sample3_pvalues],
    method="stouffer"
)
```

The `combine_p_values` function supports two methods:
- **Fisher's method**: Combines p-values using Fisher's combined probability test (default)
- **Stouffer's method**: Combines p-values using Stouffer's Z-score method

## Output

### `granp` Function Output
The `granp` function returns a Pandas DataFrame with two columns:
- `gene_names`: The identifier for each gene
- `p_values`: The p-value quantifying the statistical significance of spatial variability for each gene

### `combine_p_values` Function Output
The `combine_p_values` function returns a Pandas DataFrame with three columns:
- `gene_names`: The identifier for each gene
- `number_samples`: The number of samples/datasets where each gene was present
- `calibrated_p_values`: The combined p-value across samples using the specified method

Each row in these DataFrames represents a unique gene from the input gene expression matrix. This structured format enhances the ease of conducting sophisticated biological analyses, allowing for straightforward identification and investigation of genes with significant expression variability.

## Reference
- Li, Jinpu, Yiqing Wang, Mauminah Azam Raina, Chunhui Xu, Li Su, Qi Guo, Qin Ma, Juexin Wang, and Dong Xu. "scBSP: A fast and accurate tool for identifying spatially variable genes from spatial transcriptomic data." bioRxiv (2024).

- Wang, Juexin, Jinpu Li, Skyler T. Kramer, Li Su, Yuzhou Chang, Chunhui Xu, Michael T. Eadon, Krzysztof Kiryluk, Qin Ma, and Dong Xu. "Dimension-agnostic and granularity-based spatially variable gene identification using BSP." Nature Communications 14, no. 1 (2023): 7367.
