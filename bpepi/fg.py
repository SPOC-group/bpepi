import numpy as np
from st import SparseTensor, compute_Lambdas, compute_Lambdas_vec, compute_Lambdas_fullvec

class FactorGraph:
    """Class to update the BP messages for the SI model"""

    def __init__(self, N, T, contacts, obs, delta):
        """Construction of the FactorGraph object, starting from contacts and observations

        Args:
            N (int): Number of nodes in the contact network
            T (int): Time at which the simulation stops
            contacts (list): List of all the contacts, each given by a list (i, j, t, lambda_ij(t) )
            obs (list): List of the observations, each given by a list (i, 0/1, t), where 0 corresponds to S and 1 to I
        """
        self.messages = SparseTensor(N, T, contacts)
        self.size = N
        self.time = T
        self.delta = delta
        self.contacts = contacts

        self.Lambda0 = SparseTensor(Tensor_to_copy=self.messages) #messages only depend on lambda through Lambda matrices.
        self.Lambda1 = SparseTensor(Tensor_to_copy=self.messages)

        #compute_Lambdas(self.Lambda0, self.Lambda1, contacts)
        #compute_Lambdas_vec(self.Lambda0, self.Lambda1, contacts)
        compute_Lambdas_fullvec(self.Lambda0, self.Lambda1, contacts)

        self.observations = np.ones((N,T+2))  #creating the mask for observations
        for o in obs:
            if o[1] == 1:
                self.observations[o[0]][o[2]+1:] = 0 
            if o[1] == 0:
                self.observations[o[0]][:o[2]+1] = 0

    def iterate(self):
        """Single iteration of the BP algorithm

        Returns:
            difference (float): Maximum difference between the messages at two consecutive iterations
        """
        T=self.time
        old_msgs = SparseTensor(Tensor_to_copy=self.messages, Which=1)
        order_nodes = np.arange(0,self.size) 
        for i in order_nodes:
            inc_indices, out_indices = old_msgs.get_all_indices(i)  
            inc_msgs = old_msgs.get_neigh_i(i) 
            inc_lambda0 = self.Lambda0.get_neigh_i(i)
            inc_lambda1 = self.Lambda1.get_neigh_i(i)
            order_edges = np.arange(0,len(out_indices))
            for j in order_edges: 
                idx = out_indices[j]
                inc_msgs_j = np.delete(inc_msgs, j, axis=0)
                inc_lambda0_j = np.delete(inc_lambda0, j, axis=0)
                inc_lambda1_j = np.delete(inc_lambda1, j, axis=0)
                gamma0_ki_j = np.reshape(np.prod(np.sum(inc_lambda0_j*inc_msgs_j,axis=1),axis=0),(1,T+2))
                gamma1_ki_j = np.reshape(np.prod(np.sum(inc_lambda1_j*inc_msgs_j,axis=1),axis=0),(1,T+2))
                self.messages.values[idx] = np.transpose(((1-self.delta)*np.reshape(self.observations[i],(1,T+2))*(inc_lambda1[j]*gamma1_ki_j - inc_lambda0[j]*gamma0_ki_j)))
                self.messages.values[idx][0] = self.delta*self.observations[i][0]*np.prod(np.sum(inc_msgs_j[:,:,0],axis=1),axis=0)
                self.messages.values[idx][T+1] = np.transpose((1-self.delta)*self.observations[i][T+1]*inc_lambda1[j][:,T+1]*gamma1_ki_j[0][T+1])
                norm = self.messages.values[idx].sum() #normalize the messages
                self.messages.values[idx] = self.messages.values[idx]/norm

        difference = np.abs(old_msgs.values - self.messages.values).max()

        return difference

    def pop_dyn_RRG(self, c=3):
        """Single iteration of the Population dynamics algorithm for a d-RRG

        Args:
            c (int): degree of the RRG

        Returns:
            difference (float): Maximum difference between the messages at two consecutive iterations
        """
        T = self.time
        N = self.size
        old_msgs = SparseTensor(Tensor_to_copy=self.messages, Which=1)
        for i in np.range(N):
            indices = [np.randint(0,N) for _ in range(c-1)]
            inc_msgs = [old_msgs[idx] for idx in indices]
            inc_lambda0 = [self.Lambda0[idx] for idx in indices]
            inc_lambda1 = [self.Lambda1[idx] for idx in indices]
            gamma0_ki = np.reshape(np.prod(np.sum(inc_lambda0*inc_msgs,axis=1),axis=0),(1,T+2))
            gamma1_ki = np.reshape(np.prod(np.sum(inc_lambda1*inc_msgs,axis=1),axis=0),(1,T+2))
            self.messages.values[i] = np.transpose(((1-self.delta)*np.reshape(self.observations[i],(1,T+2))*(inc_lambda1[0]*gamma1_ki - inc_lambda0[0]*gamma0_ki)))
            self.messages.values[i][0] = self.delta*self.observations[i][0]*np.prod(np.sum(inc_msgs[:,:,0],axis=1),axis=0)
            self.messages.values[i][T+1] = np.transpose((1-self.delta)*self.observations[i][T+1]*inc_lambda1[0][:,T+1]*gamma1_ki[0][T+1])
            norm = self.messages.values[i].sum() #normalize the messages
            self.messages.values[i] = self.messages.values[i]/norm

        difference = np.abs(old_msgs.values - self.messages.values).max()

        return difference
        
    def update(self, maxit=100, tol=1e-6):
        """Multiple iterations of the BP algorithm through the method iterate()

        Args:
            maxit (int, optional): Maximum number of iterations of BP. Defaults to 100.
            tol (double, optional): Tolerance threshold for the difference between consecutive. Defaults to 1e-6.

        Returns:
            i (int): Iteration at which the algorithm stops
            error (float): Error on the messages at the end of the iterations
        """
        i = 0
        error = 1
        while i<maxit and error>tol:
            error = self.iterate()
            i+=1
        return i, error


    def marginals(self):
        """Computes the array of the BP marginals for each node

        Returns:
            marginals (np.array): Array of the BP marginals, of shape N x (T+2)
        """

        marginals = []
        #we have one marginal (Tx1 vector) for each node.
        for n in range(self.size): #loop through all nodes
            inc_indices, out_indices = self.messages.get_all_indices(n)
            inc_msg = self.messages.values[inc_indices[0]] #b_i(t_i) is the same regardless of which non directed edge (ij), j \in\partial i we pick, so long as we sum over j. 
            out_msg = self.messages.values[out_indices[0]]
            marg = np.sum(inc_msg*np.transpose(out_msg), axis=0) #transpose outgoing message so index to sum over after broadcasting is 0.
            marginals.append(marg/marg.sum())
        return np.asarray(marginals)

    def loglikelihood(self): 
        """Computes the LogLikelihood from the BP messages

        Returns:
            logL (float): LogLikelihood 
        """
        T=self.time

        log_zi = 0.
        dummy_array = np.zeros((1,T+2))
        for i in range(self.size):
            inc_indices, out_indices = self.messages.get_all_indices(i) 
            inc_msgs = self.messages.get_neigh_i(i)
            inc_lambda0 = self.Lambda0.get_neigh_i(i)
            inc_lambda1 = self.Lambda1.get_neigh_i(i)

            gamma0_ki = np.reshape(np.prod(np.sum(inc_lambda0*inc_msgs,axis=1),axis=0),(1,T+2))
            gamma1_ki = np.reshape(np.prod(np.sum(inc_lambda1*inc_msgs,axis=1),axis=0),(1,T+2))
            dummy_array = np.transpose(((1-self.delta)*np.reshape(self.observations[i],(1,T+2))*(gamma1_ki - gamma0_ki)))
            dummy_array[0] = self.delta*self.observations[i][0]*np.prod(np.sum(inc_msgs[:,:,0],axis=1),axis=0)
            dummy_array[T+1] = np.transpose((1-self.delta)*self.observations[i][T+1]*gamma1_ki[0][T+1])
            log_zi = log_zi + np.log(dummy_array.sum() )

        log_zij = 0.
        #we have one marginal (Tx1 vector) for each node.
        for n in range(self.size): #loop through all nodes
            inc_indices, out_indices = self.messages.get_all_indices(n)
            for j in np.arange(0,len(out_indices)):
                inc_msg = self.messages.values[inc_indices[j]] #b_i(t_i) is the same regardless of which non directed edge (ij), j \in\partial i we pick, so long as we sum over j. 
                out_msg = self.messages.values[out_indices[j]]
                marg = np.sum(inc_msg*np.transpose(out_msg), axis=0) #transpose outgoing message so index to sum over after broadcasting is 0.
                log_zij = log_zij + 0.5*np.log(marg.sum())
        return log_zi - log_zij

    def reset_obs(self, obs):
        """Resets the observations, starting from the obs list
        
        Args:
            obs (list): List of the observations, each given by a list (i, 0/1, t), where 0 corresponds to S and 1 to I

        """
        self.observations = np.ones((self.size,self.time+2))  #creating the mask for observations
        for o in obs:
            if o[1] == 1:
                self.observations[o[0]][o[2]+1:] = 0 
            if o[1] == 0:
                self.observations[o[0]][:o[2]+1] = 0
