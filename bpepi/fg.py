import numpy as np
from st import SparseTensor, compute_Lambdas

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

        compute_Lambdas(self.Lambda0, self.Lambda1, contacts)

        self.observations = np.ones((N,T+2))  #creating the mask for observations
        for o in obs:
            if o[1] == 1:
                self.observations[o[0]][o[2]+1:] = 0 
            if o[1] == 0:
                self.observations[o[0]][:o[2]+1] = 0

        max_l = len(max(self.messages.adj_list,key=len))-1  #maximum degree of graph
        flat_adj = [i for sublist in self.messages.adj_list for i in sublist]
        length = len(flat_adj) #number of edges
        self.out_msgs = np.array([-1]) #array of outgoing messages to update
        self.inc_msgs = [] #2D array with self.inc_msgs[i] = array of incoming messages needed to update message self.out_msgs[i] minus reversed edge
        self.inc_j = [] #self.inc_j[i] = reversed edge of self.out_msgs[i]

        for i in range(len(self.messages.idx_list)):
            self.out_msgs = np.concatenate((self.out_msgs,self.messages.idx_list[i]),axis=0)
            for j in range(len(self.messages.idx_list[i])):
                k = self.messages.adj_list[i][j]
                jnd = np.where(np.array(self.messages.adj_list[k])==i)[0]
                ind = np.where(np.array(self.messages.adj_list[k])!=i)[0]
                self.inc_j.append(self.messages.idx_list[k][jnd])
                self.inc_msgs.append(np.concatenate((self.messages.idx_list[k][ind],np.full(max_l-len(ind), length)))) #to all that have < max neighbours, throw missing neighbours to fictitious message and lambda matrix.

        self.out_msgs = np.delete(self.out_msgs, 0)
        self.inc_msgs = np.array(self.inc_msgs)
        self.inc_j = np.array(self.inc_j).flatten()
        self.obs_i = np.array(flat_adj)

        self.Lambda0.values = np.append(self.Lambda0.values,np.full((1,T+2,T+2),1/(T+2)**2),axis=0) #add fictitious lamda matrices
        self.Lambda1.values = np.append(self.Lambda1.values,np.full((1,T+2,T+2),1/(T+2)**2),axis=0)

    def iterate(self): #only works all types of graphs, not just RRG?
        """Single iteration of the BP algorithm
        Returns:
            difference (float): Maximum difference between the messages at two consecutive iterations
        """
        T=self.time
        old_msgs = SparseTensor(Tensor_to_copy=self.messages, Which=1)
        old_msgs.values = np.append(old_msgs.values,np.full((1,T+2,T+2),1/(T+2)**2),axis=0) #add fictitious message which contributes nothing in update
        gamma0 = np.reshape(np.prod(np.sum(self.Lambda0.values[self.inc_msgs]*old_msgs.values[self.inc_msgs],axis=2),axis=1),(len(self.inc_msgs),1,T+2))
        gamma1 = np.reshape(np.prod(np.sum(self.Lambda1.values[self.inc_msgs]*old_msgs.values[self.inc_msgs],axis=2),axis=1),(len(self.inc_msgs),1,T+2))
        one = np.transpose((1-self.delta)*np.reshape(self.observations[self.obs_i],(len(self.out_msgs),1,T+2))*(self.Lambda1.values[self.inc_j]*gamma1 - self.Lambda0.values[self.inc_j]*gamma0),(0,2,1))[:,1:T+1,:]
        two = np.reshape(np.tile(np.reshape(self.delta*self.observations[self.obs_i][:,0]*np.prod(np.sum(old_msgs.values[self.inc_msgs][:,:,:,0],axis=2),axis=1),(len(self.out_msgs),1)),T+2),(len(self.out_msgs),1,T+2))
        three = np.reshape((1-self.delta)*np.reshape(self.observations[self.obs_i][:,T+1],(len(self.out_msgs),1))*self.Lambda1.values[self.inc_j][:,:,T+1]*np.reshape(gamma1[:,0,T+1],(len(self.out_msgs),1)),(len(self.out_msgs),1,T+2))
        self.messages.values[self.out_msgs] = np.concatenate((np.zeros((len(self.out_msgs),1,T+2)),one,np.zeros((len(self.out_msgs),1,T+2))), axis=1) + np.concatenate((two,np.zeros((len(self.out_msgs),T+1,T+2))),axis=1) + np.concatenate((np.zeros((len(self.out_msgs),T+1,T+2)),three),axis=1)
        norm = np.reshape(np.sum(self.messages.values,axis=(1,2)),(len(self.out_msgs),1,1)) #normalize the messages
        self.messages.values = self.messages.values/norm
        old_msgs.values = np.delete(old_msgs.values,len(self.out_msgs),axis=0) #remove fictitious message to calculate difference
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
        for i in range(N):
            indices = [np.random.randint(0,N) for _ in range(c-1)]
            inc_msgs = np.array([old_msgs.values[idx] for idx in indices])
            inc_lambda0 = np.array([self.Lambda0.values[idx] for idx in indices])
            inc_lambda1 = np.array([self.Lambda1.values[idx] for idx in indices])
            gamma0_ki = np.reshape(np.prod(np.sum(inc_lambda0*inc_msgs,axis=1),axis=0),(1,T+2))
            gamma1_ki = np.reshape(np.prod(np.sum(inc_lambda1*inc_msgs,axis=1),axis=0),(1,T+2))
            self.messages.values[i] = np.transpose(((1-self.delta)*np.reshape(np.ones(T+2),(1,T+2))*(inc_lambda1[0]*gamma1_ki - inc_lambda0[0]*gamma0_ki)))
            self.messages.values[i][0] = self.delta*np.prod(np.sum(inc_msgs[:,:,0],axis=1),axis=0)
            self.messages.values[i][T+1] = np.transpose((1-self.delta)*inc_lambda1[0][:,T+1]*gamma1_ki[0][T+1])
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
                log_zij = log_zij + np.log(marg.sum())
        return log_zi - 0.5 * log_zij

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
