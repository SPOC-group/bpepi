# BPEPI (Belief Propagation for EPidemic Inference)
  
Belief propagation algorithm for inferring patient zeros in compartmental models.

Written by Antoine Aragon, Davide Ghio, Indaco Biazzo and Lenka Zdeborová. Last updated on 23-11-2022.

## Requirements

- `python3`
- `numpy`
- `networkx`

## Installation

run `python setup.py install`

## Documentation

For example usage see the `examples/demo.ipynb` notebook.

### FactorGraph class

```
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

        Class to update the BP messages for the SI model

        Arguments
        ---------
        N (int): Number of nodes in the contact network
        T (int): Time at which the simulation stops
        contacts (list): List of all the contacts, each given by a list (i, j, t, lambda_ij(t))
        obs (list): List of the observations, each given by a list (i, 0/1, t), where 0 corresponds to S and 1 to I
        delta (float): Probability for an individual to be a source
        mask (list): if it is equal to ["SI"], the function simulates an SI model, otherwise the i-th element of the list (between 0 and 1) represents the infectivity of the nodes at i timesteps after the infection
        mask_type (string): Type of inference model. If equal to "SIR", it means we are simulating a SIR model and inferring using the dSIR model

        Main methods
        ------------
        iterate(damp): Single iteration of the Belief Propagation algorithm
        update(maxit, tol, damp, print_iter): Multiple iterations of the BP algorithm through the iterate method
        marginals(): Computes the array of the BP marginals for each node
        reset_obs(obs): Resets the observations, starting from a provided list.
```

### SparseTensor class

```
SparseTensor(
    N = 0,
    T = 0,
    contacts = [],
    Tensor_to_copy = None
)

        Class to represent an N x N x T x T sparse tensor as a 2 x num_edges x T x T full tensor

        Arguments
        ---------
        N (int): Number of nodes in the contact network
        T (int): Value of the last simulation time
        contacts (list): List of all the contacts, each given by a list (i, j, t, lambda_ij(t))
        Tensor_to_copy (SparseTensor): An existing SparseTensor to copy and from which to create a new object
        
        Main methods
        ------------
        init_like(Tensor): Initialization of the tensor, given another tensor, putting all values to one
        get_idx_ij(i, j): Returns index corresponding to the (i, j) entrance of the tensor
        get_ij(i, j): Returns the T x T matrix corresponding to the (i, j) entrance of the tensor
        get_neigh_i(i): Returns d_i T x T matrices corresponding to the (*, i) entrances of the tensor
        get_all_indices(i): Returns all the indices corresponding to the incoming and outgoing messages for a given node
```

