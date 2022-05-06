import numpy as np

class SparseTensor:
    """Class to represent an N x N x T x T sparse tensor as a 2 x num_edges x T x T full tensor"""
    def __init__(self) -> None:
        pass

    def init(self, N, T, contacts, fill_value = 0.):
        """Initialization of the tensor, given the contacts

        Args:
            N (int): Number of nodes in the contact network
            T (int): Value of the last simulation time
            contacts (np.array): Array of all the contacts, each given by a list (i, j, t, lambda_ij)
        """
        self.idx_list = []
        self.adj_list = [ [] for _ in range(N) ]

        self.N = N
        self.T = T

        edge_list = np.unique(np.asarray(contacts, dtype = 'int')[:,:2],axis=0) #We get the contact network directly from the list of contacts
        self.num_direct_edges=len(edge_list)

        for e in edge_list:
            self.adj_list[e[0]].append(e[1])
        self.degree = np.array(len(a) for a in self.adj_list)
        c = 0
        for d in self.degree:
            self.idx_list.append(np.arange(c,c+d))
            c = c + d

        self.values = np.full((self.num_direct_edges, T+1, T+1), fill_value)

    def init_like(self, Tensor, fill_value=0.):
        """Initialization of the tensor, given another tensor

        Args:
            Tensor (SparseTensor): The Sparse Tensor object we want to copy
        """
        self.idx_list = Tensor.idx_list
        self.adj_list = Tensor.adj_list
        self.N = Tensor.N
        self.T = Tensor.T
        self.num_direct_edges=Tensor.num_direct_edges
        self.degree = Tensor.degree
        self.values = np.full((self.num_direct_edges, self.T+1, self.T+1), fill_value)

    def get_ij(self, i, j):
        """Returns the T x T matrix corresponding to the (i, j) entrance of the tensor

        Args:
            i (int): Index of the sending node
            j (int): Index of the receiving node
        """
        idx_i = self.adj_list[j].index(i)
        idx = self.idx_list[j][idx_i]

        return self.values[idx]
        
    def get_neigh_i(self, i):
        """Returns d_i T x T matrices corresponding to the (*, i) entrances of the tensor

        Args:
            i (int): Index of the receiving node
        """
        return self.values[self.idx_list[i]]