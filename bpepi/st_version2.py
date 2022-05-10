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
        self.lambda_list = [ [] for _ in range(N) ] #also add a list with same format as adj_list.

        self.N = N
        self.T = T
        edge_list = np.unique(np.delete(np.asarray(contacts), 2, axis = 1), axis = 0)[:,:2].astype('int') #We get the contact network directly from the list of contacts
        lam = np.unique(np.delete(np.asarray(contacts), 2, axis = 1), axis = 0)[:,2] #also gather the lambdas which depend on the edges
        self.num_direct_edges=len(edge_list)

        for e in range(len(edge_list)):
            self.adj_list[edge_list[e][0]].append(edge_list[e][1])
            self.lambda_list[edge_list[e][0]].append(lam[e])
        self.degree = np.array(len(a) for a in self.adj_list)
        c = 0
        for d in self.degree.tolist():
            self.idx_list.append(np.arange(c,c+d).tolist())
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
        
    def get_inc_i(self, i):
        """Returns d_i T x T matrices corresponding to the (*, i) entrances of the tensor

        Args:
            i (int): Index of the receiving
        """
        return self.values[self.idx_list[i]]

    def get_indices(self, i):
        """Returns indices of messages outoing from node i

        Args:
            i (int): Index of the sending node
        """
        idx = []  #first index in idx_list & adj_list represents the node recieving the message from the second index. 
                  #So for outgoing messages, we need to do the same thing as in get_ij
        for j in self.adj_list[i]:
            idx_i = self.adj_list[j].index(i)
            idx.append(self.idx_list[j][idx_i])

        return idx