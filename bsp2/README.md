# bsp2
bsp2 is a specialized package designed for processing biological data, specifically in the analysis of gene expression and cell coordinates. It efficiently computes p-values for a given set of genes based on input matrices representing cell coordinates and gene expression data.

## Installation
`pip install "git+https://github.com/YQ-Wang/bsp2.git"`

## Usage

To use bsp2, you need to provide two primary inputs:

1. **Cell Coordinates Matrix (`input_sp_mat`)**: 
   - Format: Numpy array.
   - Dimensions: N x D, where N is the number of cells and D is the dimension of coordinates.

2. **Gene Expression Matrix (`input_exp_mat_raw`)**:
   - Format: Numpy array, Pandas DataFrame, or CSR matrix.
   - Dimensions: N x P, where N is the number of cells and P is the number of genes.

Additionally, you must specify the following parameters:

- `d1`: A floating-point number.
- `d2`: A floating-point number.

### Example

```python
import bsp2

# Example data loading
input_sp_mat = ...
input_exp_mat_raw = ...

# Optional parameters
d1 = ...
d2 = ...

# Calculate p-values
p_values = bsp2.granp(input_sp_mat, input_exp_mat_raw, d1, d2)
```

## Output

The output of bsp2 is a list of p-values corresponding to the given genes.
