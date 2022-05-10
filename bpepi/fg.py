import numpy as np

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
        self.messages = SparseTensor() #create abstract messages object
        self.messages.init(N, T, contacts) #intialize the messages
        self.size = N
        self.time = T
        self.delta = delta
        self.contacts = contacts

        self.lambda0 = SparseTensor() #messages only depend on lambda through Lambda matrices.
        self.lambda0.init(N, T, contacts)
        self.lambda1 = SparseTensor()
        self.lambda1.init(N, T, contacts)

        for i, e in enumerate(self.messages.adj_list): #Go through each direct edge, initialize messages and calculate lambda matrices
            for j in range(len(e)): 
                idx = self.messages.idx_list[i][j] #message index that corresponds to a given direct edge from adj list. 
                msg = np.random.uniform(low = 0.0, high = 1.0, size = (T+1, T+1))
                msg = msg/msg.sum()
                self.messages.values[idx] = msg #initialize the value of the messages using mapping adj_list -> idx_list mapping
                lam_ij = self.lambda0.lambda_list[i][j] #lambda value for edge (should be the same for lambda1 and lambda0)
                lambda0_ij = np.zeros((T+1,T+1))
                lambda1_ij = np.zeros((T+1,T+1))
                for ti in range(T+1):
                    for tj in range(T+1):
                        lambda0_ij[ti][tj] = (1-lam_ij)**((ti-tj)*np.heaviside(ti-tj, 0))
                        lambda1_ij[ti][tj] = (1-lam_ij)**((ti-tj-1)*np.heaviside(ti-tj, 0))
                self.lambda0.values[idx] = lambda0_ij #use adj_list -> idx_list mapping to initialize lambdas
                self.lambda1.values[idx] = lambda1_ij

        self.observations = np.zeros((N,T+1))  #creating the mask for observations
        for i in range(N):
            positions = np.where(obs[:,0]==i)[0] 
            if len(positions) == 0:
                self.observations[i] = np.ones((1,T+1))
            if len(positions) >= 1:
                self.observations[obs[positions[-1]][0]][obs[positions[-1]][2]:] = np.ones((1,T+1-obs[positions[-1]][2]))

    def update(self, maxit=100, tol=1e-6):
        """_summary_

        Args:
            maxit (int, optional): _description_. Defaults to 100.
            tol (double, optional): _description_. Defaults to 1e-6.
        """
        i = 0
        while i<maxit:
            error = self.iterate()
            i+=1
        return error

    def iterate(self): #update all the messages once, return the maximum difference between any message and its value before being updated.
        """_summary_
        
        Args:
        """
        old_msgs = self.messages.values
        for i in range(len(self.messages.adj_list)):
            self.update_msg(i)
        new_msgs = self.messages.values

        difference = []
        for m in range(len(self.messages.values)):
            difference.append(np.linalg.norm(new_msgs[m]-old_msgs[m]))

        return max(difference)
        
    def update_msg(self, i):
        """_summary_

        Args:
            i (_type_): _description_
        """
        #want to update all messages that come out of node i. 
        inc_msgs = self.messages.get_inc_i(i)
        inc_lambda0 = self.lambda0.get_inc_i(i)
        inc_lambda1 = self.lambda1.get_inc_i(i)

        #now find the outgoing messages. Code relies on self.messages.get_indices being in same order as messages in self.lambda0.get_inc_i.
        #ie. for d=3:  self.lambda0.get_inc_i = [(1,2),(4,2),(7,2)] and self.messages.get_indices = [(2,1),(2,4),(2,7)].
        updated_msgs = []
        for j in range(len(self.messages.get_indices(i))):
            idx = self.messages.get_indices(i)[j]
            no_j = np.delete(inc_msgs, j, axis=0)
            gamma0_ki_j = np.prod(np.einsum('ijk,ijk->ik',np.delete(inc_lambda0, j, axis=0),no_j),axis=0)
            gamma1_ki_j = np.prod(np.einsum('ijk,ijk->ik',np.delete(inc_lambda1, j, axis=0),no_j),axis=0)
            self.messages.values[idx][1:] = ((1-self.delta)*np.transpose(self.observations[i])*(inc_lambda1[j]*gamma1_ki_j - inc_lambda0[j]*gamma0_ki_j))[1:]
            self.messages.values[idx][0] = self.delta*np.transpose(self.observations[i])*np.prod(np.einsum('ijk->ij',no_j),axis=0)[0]
            norm = np.sum(self.messages.values[idx]) #normalize the messages
            self.messages.values[idx] = self.messages.values[idx]/norm
            updated_msgs.append(self.messages.values[idx])

    def marginals(self): #calculate marginals for infection times.
        """_summary_"""

        adj = self.messages.adj_list
        marginals = []
        #we have one marginal (Tx1 vector) for each node.
        for n in range(len(adj)): #loop through all nodes
            m1 = self.messages.values[adj[n][0]]
            idx = self.messages.get_indices(n)
            m2 = self.messages.values[idx[0]]
            marg = np.einsum('ij,ji->i',m1,m2) #m1 has to be first, since m1 = m1(t_i,t_j). 
            marg = marg/marg.sum()
            marginals.append(marg)
        return np.asarray(marginals)
          