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

    def iterate(self):
        old_msgs = SparseTensor(Tensor_to_copy=self.messages, Which=1)
        order_nodes = np.arange(0,self.size) 
        np.random.shuffle(order_nodes) #shuffle order in which nodes are updated.
        for i in order_nodes:
            inc_indices, out_indices = old_msgs.get_all_indices(i)
            inc_msgs = old_msgs.get_neigh_i(i)
            inc_lambda0 = self.Lambda0.get_neigh_i(i)
            inc_lambda1 = self.Lambda1.get_neigh_i(i)
            order_edges = np.arange(0,len(out_indices)) #shuffle order in which outoing messages are updated for each node.
            np.random.shuffle(order_edges)
            for j in order_edges: 
                idx = out_indices[j]
                inc_msgs_j = np.delete(inc_msgs, j, axis=0)
                inc_lambda0_j = np.delete(inc_lambda0, j, axis=0)
                inc_lambda1_j = np.delete(inc_lambda1, j, axis=0)
                gamma0_ki_j = np.reshape(np.prod(np.sum(inc_lambda0_j*inc_msgs_j,axis=1),axis=0),(1,T+2))
                gamma1_ki_j = np.reshape(np.prod(np.sum(inc_lambda1_j*inc_msgs_j,axis=1),axis=0),(1,T+2))
                self.messages.values[idx] = np.transpose(((1-self.delta)*np.reshape(self.observations[i],(1,T+2))*(inc_lambda1[j]*gamma1_ki_j - inc_lambda0[j]*gamma0_ki_j)))
                self.messages.values[idx][0] = self.delta*self.observations[i][0]*np.prod(np.sum(inc_msgs_j,axis=1),axis=0)[0]
                self.messages.values[idx][T+1] = np.transpose((1-self.delta)*self.observations[i][T+1]*inc_lambda1[j][:,T+1]*gamma1_ki_j[0][T+1])
                norm = self.messages.values[idx].sum() #normalize the messages
                self.norms.append(norm)
                self.messages.values[idx] = self.messages.values[idx]/norm

        difference = np.abs(old_msgs.values - self.messages.values).max()

        return difference
    
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
    
    def marginals(self): #calculate marginals for infection times.
        """_summary_"""

        idx_list = self.messages.idx_list
        marginals = []
        #we have one marginal (Tx1 vector) for each node.
        for n in range(self.size): #loop through all nodes
            inc_indices, out_indices = self.messages.get_all_indices(n)
            inc_msg = self.messages.values[inc_indices[0]] #b_i(t_i) is the same regardless of which non directed edge (ij), j \in\partial i we pick, so long as we sum over j. 
            out_msg = self.messages.values[out_indices[0]]
            marg = np.sum(inc_msg*np.transpose(out_msg), axis=0) #transpose outgoing message so index to sum over after broadcasting is 0.
            marginals.append(marg/marg.sum())
        return np.asarray(marginals)
        
