# scBSP

scBSP is a specialized package designed for processing biological data, specifically in the analysis of gene expression and cell coordinates. It efficiently computes p-values for a given set of genes based on input matrices representing cell coordinates and gene expression data.

## Installation

To install scBSP, run the following command:

`pip install scbsp`

## Usage

To use scBSP, you need to provide two primary inputs:

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

Here's a simple example to demonstrate how to compute p-values using scBSP:

```python
import scbsp

# Load your data into these variables
input_sp_mat = ...  # Cell Coordinates Matrix
input_exp_mat_raw = ...  # Gene Expression Matrix

# Set the optional parameters
d1 = 0.5  # Example value
d2 = 0.5  # Example value

# Execute the calculation
p_values = scbsp.granp(input_sp_mat, input_exp_mat_raw, d1, d2)
```

## Output

The function returns a list of p-values, each corresponding to the genes in the provided gene expression matrix. These p-values help in identifying significant differences in gene expression across different cell coordinates, facilitating advanced biological data analysis.
