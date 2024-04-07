# scBSP

scBSP is a dedicated software package crafted for the nuanced domain of biological data processing, emphasizing gene expression analysis and cell coordinate evaluation. It offers a streamlined method to calculate p-values for a set of genes by leveraging input matrices that encapsulate cell coordinates and gene expression data.

## Installation

### Dependencies
To ensure scBSP functions optimally, the following dependencies are required:
- Python (>= 3.8)
- NumPy (>= 1.24.4)
- Pandas (>= 1.3.5)
- SciPy (>= 1.10.1)
- scikit-learn (>=1.3.2)

For enhanced scBSP using HNSW for distance calculation:
- hnswlib (>= 0.8.0)

### Installation Commands
For Standard Installation (Using Ball Tree):

`pip install scbsp`

For Installation with HNSW (Hierarchical Navigable Small World Graphs):

`pip install scbsp[hnsw]`

## Usage

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
- `leaf_size`: Optional integer defining the maximum point threshold for the Ball Tree algorithm to revert to brute-force search (default = 80). Not required for installations using HNSW.


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

## Output

The function returns a list of p-values, each corresponding to the genes in the provided gene expression matrix. These p-values help in identifying significant differences in gene expression across different cell coordinates, facilitating advanced biological data analysis.
