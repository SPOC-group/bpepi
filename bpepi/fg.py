import numpy as np
from st import SparseTensor, compute_Lambdas

class FactorGraph:
    """_summary_"""

    def __init__(self, N, T, contacts, obs, delta):
        """_summary_

        Args:
            N (_type_): _description_
            T (_type_): _description_
            contacts (_type_): _description_
            observations (_type_): _description_
        """
        self.messages = SparseTensor(N, T, contacts)
        self.size = N
        self.time = T
        self.delta = delta
        self.contacts = contacts

        self.Lambda0 = SparseTensor(Tensor_to_copy=self.messages) #messages only depend on lambda through Lambda matrices.
        self.Lambda1 = SparseTensor(Tensor_to_copy=self.messages)

        compute_Lambdas(self.Lambda0, self.Lambda1, contacts)

        self.observations = np.ones((N,T+1))  #creating the mask for observations
        for o in obs:
            if o[1].pi == 1:
                self.observations[o[0]][o[2]+1:] = 0 
            if o[1].ps == 1:
                self.observations[o[0]][:o[2]+1] = 0


    def update(self, maxit=100, tol=1e-6):
        """_summary_

        Args:
            maxit (int, optional): _description_. Defaults to 100.
            tol (double, optional): _description_. Defaults to 1e-6.
        """
        i = 0
        error = 1
        while i<maxit and error>tol:
            error = self.iterate()
            i+=1
        return i, error

    def iterate(self): #update all the messages once, return the maximum difference between any message and its value before being updated.
        """_summary_
        
        Args:
        """
        old_msgs = self.messages.values
        for i in range(len(self.messages.adj_list)):
            self.update_msg(i)
        new_msgs = self.messages.values

        difference = np.abs(old_msgs - new_msgs).max()

        return difference
        
    def update_msg(self, i): #update all messages that come out of node i. 
        """_summary_

        Args:
            i (_type_): _description_
        """
        inc_indices, out_indices = self.messages.get_all_indices(i)
        inc_msgs = self.messages.get_neigh_i(i)
        inc_lambda0 = self.Lambda0.get_neigh_i(i)
        inc_lambda1 = self.Lambda1.get_neigh_i(i)

        for j in range(len(out_indices)):
            idx = out_indices[j]
            inc_msgs_j = np.delete(inc_msgs, j, axis=0)
            inc_lambda0_j = np.delete(inc_lambda0, j, axis=0)
            inc_lambda1_j = np.delete(inc_lambda1, j, axis=0)
            gamma0_ki_j = np.reshape(np.prod(np.sum(inc_lambda0_j*inc_msgs_j,axis=1),axis=0),(1,T+1))
            gamma1_ki_j = np.reshape(np.prod(np.sum(inc_lambda1_j*inc_msgs_j,axis=1),axis=0),(1,T+1))
            self.messages.values[idx] = np.transpose(((1-self.delta)*np.reshape(self.observations[i],(1,T+1))*(inc_lambda1[j]*gamma1_ki_j - inc_lambda0[j]*gamma0_ki_j)))
            self.messages.values[idx][0] = self.delta*self.observations[i][0]*np.prod(np.sum(inc_msgs_j,axis=1),axis=0)[0]
            norm = self.messages.values[idx].sum() #normalize the messages
            self.messages.values[idx] = self.messages.values[idx]/norm

    def marginals(self): #calculate marginals for infection times.
        """_summary_"""

        idx_list = self.messages.idx_list
        marginals = []
        #we have one marginal (Tx1 vector) for each node.
        for n in range(len(idx_list)): #loop through all nodes
            idx = self.messages.get_inc_indices(n)
            m1 = self.messages.values[idx[0]]
            m2 = self.messages.values[idx_list[n][0]]
            marg = np.sum(m2*np.transpose(m1), axis = 1) #m1 is transposed since m1 is the incoming message 
            marg = marg/marg.sum()
            marginals.append(marg)
        return np.asarray(marginals)