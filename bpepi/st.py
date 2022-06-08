import numpy as np
import copy


class SparseTensor:
    """Class to represent an N x N x T x T sparse tensor as a 2 x num_edges x T x T full tensor"""

    def __init__(self, N=0, T=0, contacts=[], Tensor_to_copy=None, Which=None):
        """Construction of the tensor. If no tensor is given, then calls init(), otherwise calls init_like()

        Args:
            N (int): Number of nodes in the contact network
            T (int): Value of the last simulation time
            contacts (list): List of all the contacts, each given by a list (i, j, t, lambda_ij(t) )
            Tensor_to_copy (SparseTensor): SparseTensor to copy to create a new object
        """
        if Tensor_to_copy is None:
            self.init(N, T, contacts)
        else:
            if Which is None:
                self.init_like(Tensor_to_copy)
            else:
                self.init_like2(Tensor_to_copy)

    def init(self, N, T, contacts):
        """Initialization of the tensor, given the contacts

        Args:
            N (int): Number of nodes in the contact network
            T (int): Value of the last simulation time
            contacts (list): List of all the contacts, each given by a list (i, j, t, lambda_ij(t) )
        """
        self.idx_list = []
        self.adj_list = [[] for _ in range(N)]

        self.N = N
        self.T = T

        edge_list = np.unique(
            np.asarray(contacts, dtype="int")[:, :2], axis=0
        )  # We get the contact network directly from the list of contacts
        edge_list = np.concatenate((edge_list, np.flip(edge_list, 1)), axis=0)
        edge_list = np.unique(edge_list, axis=0)
        self.num_direct_edges = len(edge_list)

        for e in edge_list:
            self.adj_list[e[0]].append(e[1])
        self.degree = np.array([len(a) for a in self.adj_list])
        c = 0
        for d in self.degree:
            self.idx_list.append(np.arange(c, c + d))
            c = c + d

        self.values = np.full(
            (self.num_direct_edges, T + 2, T + 2), 1 / ((T + 2) * (T + 2))
        )

    def init_like(self, Tensor):
        """Initialization of the tensor, given another tensor

        Args:
            Tensor (SparseTensor): The Sparse Tensor object we want to copy
        """
        self.idx_list = Tensor.idx_list
        self.adj_list = Tensor.adj_list
        self.N = Tensor.N
        self.T = Tensor.T
        self.num_direct_edges = Tensor.num_direct_edges
        self.degree = Tensor.degree
        self.values = np.full((self.num_direct_edges, self.T + 2, self.T + 2), 1.0)

    def init_like2(self, Tensor):
        """Initialization of the tensor, given another tensor

        Args:
            Tensor (SparseTensor): The Sparse Tensor object we want to copy
        """
        self.idx_list = Tensor.idx_list
        self.adj_list = Tensor.adj_list
        self.N = Tensor.N
        self.T = Tensor.T
        self.num_direct_edges = Tensor.num_direct_edges
        self.degree = Tensor.degree
        self.values = copy.deepcopy(Tensor.values)

    def get_idx_ij(self, i, j):
        """Returns index corresponding to the (i, j) entrance of the tensor

        Args:
            i (int): Index of the sending node
            j (int): Index of the receiving node
        """
        idx_i = self.adj_list[j].index(i)
        idx = self.idx_list[j][idx_i]

        return idx

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

    def get_all_indices(self, i):
        incoming_indices = self.idx_list[i]
        outgoing_indices = []
        for j in self.adj_list[i]:
            idx_i = self.adj_list[j].index(i)
            idx = self.idx_list[j][idx_i]
            outgoing_indices.append(idx)

        return incoming_indices, outgoing_indices


def compute_Lambdas(Lambda0, Lambda1, contacts):  # change to loop over full contacts
    """Computes (once and for all) the entrances of the tensors Lambda0 and Lambda1, starting from the list of contacts
    Args:
        Lambda0 (SparseTensor): SparseTensor useful to update the BP messages
        Lambda1 (SparseTensor): SparseTensor useful to update the BP messages
        contacts (list): List of all the contacts, each given by a list (i, j, t, lambda_ij(t) )
    """
    T = Lambda0.T
    con = np.array(contacts)
    for time in range(T):
        contacts_T = con[np.where(con[:, 2] == time)[0]]
        c2 = contacts_T[:, 2]
        c3 = contacts_T[:, 3]
        idx = np.zeros(len(contacts_T), dtype="int")
        i = 0
        for c in contacts_T:
            idx[i] = Lambda0.get_idx_ij(int(c[0]), int(c[1]))
            i += 1
        x1 = np.tile(
            np.reshape(np.arange(0, T + 2), (1, 1, T + 2)), (len(idx), T + 2, 1)
        )
        x2 = np.tile(
            np.reshape(np.arange(0, T + 2), (1, T + 2, 1)), (len(idx), 1, T + 2)
        )
        Lambda0.values[idx] = Lambda0.values[idx] * np.reshape(
            (np.reshape((1 - c3), (c3.shape[0], 1, 1)) * np.ones((1, T + 2, T + 2)))
            ** np.heaviside(x1 - x2 - np.reshape(c2, (c2.shape[0], 1, 1)), 0),
            (len(idx), T + 2, T + 2),
        )
        Lambda1.values[idx] = Lambda1.values[idx] * np.reshape(
            (np.reshape((1 - c3), (c3.shape[0], 1, 1)) * np.ones((1, T + 2, T + 2)))
            ** np.heaviside(x1 - x2 - np.reshape(c2, (c2.shape[0], 1, 1)) - 1, 0),
            (len(idx), T + 2, T + 2),
        )
