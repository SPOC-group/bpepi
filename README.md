# BPEPI ([B]elief [P]ropagation for [EP]idemic [I]nference)

BPEPI is a Python package implementing belief propagation algorithms for inference problems in compartmental spreading models.

## Authors

Antoine Aragon, Indaco Biazzo, Davide Ghio, and Lenka Zdeborová

## Description

BPEPI provides tools for analyzing and inferring epidemic spreading patterns using belief propagation techniques. It is particularly useful for researchers and practitioners working with compartmental models such as SI (Susceptible-Infected) and SIR (Susceptible-Infected-Recovered).

## Features

- Belief propagation algorithm for SI and SIR models
- Efficient sparse tensor representation for contact networks
- Customizable observation handling
- Flexible inference model selection

## Requirements

- Python 3.x
- NumPy
- PyTorch

## Installation

To install BPEPI, navigate to the bpepi folder and run:

```
pip install .
```

## Usage

For example usage, please refer to the `examples/demo.ipynb` notebook in the repository.

## Main Classes

### FactorGraph

```python
FactorGraph(
    N,
    T,
    contacts,
    obs,
    delta,
    mask = ["SI"],
    mask_type = "SI",
    verbose = False
)
```

This class updates the BP messages for the SI model.

#### Arguments

- `N` (int): Number of nodes in the contact network
- `T` (int): Time at which the simulation stops
- `contacts` (list): List of all contacts, each given by `[i, j, t, lambda_ij(t)]`
- `obs` (list): List of observations, each given by `[i, 0/1, t]`, where 0 corresponds to S and 1 to I
- `delta` (float): Probability for an individual to be a source
- `mask` (list): If `["SI"]`, simulates an SI model; otherwise, the i-th element (between 0 and 1) represents the infectivity of nodes at i timesteps after infection
- `mask_type` (string): Type of inference model (e.g., "SI" or "SIR")
- `verbose` (bool): If True, prints additional information during execution

#### Main Methods

- `iterate(damp)`: Single iteration of the Belief Propagation algorithm
- `update(maxit, tol, damp, print_iter)`: Multiple iterations of the BP algorithm
- `marginals()`: Computes the array of BP marginals for each node
- `reset_obs(obs)`: Resets the observations, starting from a provided list

### SparseTensor

```python
SparseTensor(
    N = 0,
    T = 0,
    contacts = [],
    Tensor_to_copy = None
)
```

This class represents an N x N x T x T sparse tensor as a 2 x num_edges x T x T full tensor.

#### Arguments

- `N` (int): Number of nodes in the contact network
- `T` (int): Value of the last simulation time
- `contacts` (list): List of all contacts, each given by `[i, j, t, lambda_ij(t)]`
- `Tensor_to_copy` (SparseTensor): An existing SparseTensor to copy and from which to create a new object

#### Main Methods

- `init_like(Tensor)`: Initialization of the tensor, given another tensor, setting all values to one
- `get_idx_ij(i, j)`: Returns index corresponding to the (i, j) entrance of the tensor
- `get_ij(i, j)`: Returns the T x T matrix corresponding to the (i, j) entrance of the tensor
- `get_neigh_i(i)`: Returns d_i T x T matrices corresponding to the (*, i) entrances of the tensor
- `get_all_indices(i)`: Returns all indices corresponding to incoming and outgoing messages for a given node

## Citation

If you use BPEPI in your research, please cite the following paper:

```
@article{PhysRevE.108.044308,
  title = {Inference of epidemic dynamics from incomplete observations},
  author = {Ghio, Davide and Aragon, Antoine L. M. and Biazzo, Indaco and Zdeborová, Lenka},
  journal = {Phys. Rev. E},
  volume = {108},
  issue = {4},
  pages = {044308},
  numpages = {19},
  year = {2023},
  month = {Oct},
  publisher = {American Physical Society},
  doi = {10.1103/PhysRevE.108.044308},
  url = {https://link.aps.org/doi/10.1103/PhysRevE.108.044308}
}
```
