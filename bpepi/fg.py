import numpy as np
from bpepi.st import SparseTensor, compute_Lambdas


class FactorGraph:
    """Class to update the BP messages for the SI model"""

    def __init__(self, N, T, contacts, obs, delta, verbose=False):
        """Construction of the FactorGraph object, starting from contacts and observations

        Args:
            N (int): Number of nodes in the contact network
            T (int): Time at which the simulation stops
            contacts (list): List of all the contacts, each given by a list (i, j, t, lambda_ij(t) )
            obs (list): List of the observations, each given by a list (i, 0/1, t), where 0 corresponds to S and 1 to I
        """
        self.messages = SparseTensor(N, T, contacts)
        if verbose:
            print("Messages matrices created")

        self.size = N
        self.time = T
        self.delta = delta
        self.contacts = contacts

        self.Lambda0 = SparseTensor(
            Tensor_to_copy=self.messages
        )  # messages only depend on lambda through Lambda matrices.
        self.Lambda1 = SparseTensor(Tensor_to_copy=self.messages)

        if verbose:
            print("Lambdas matrices created")

        compute_Lambdas(self.Lambda0, self.Lambda1, contacts)

        if verbose:
            print("Lambdas matrices computed")

        self.observations = np.ones((N, T + 2))  # creating the mask for observations
        for o in obs:
            if o[1] == 1:
                self.observations[o[0]][o[2] + 1 :] = 0
            if o[1] == 0:
                self.observations[o[0]][: o[2] + 1] = 0

        if verbose:
            print("Observations array created")

        self.out_msgs = np.array([], dtype="int")
        self.inc_msgs = np.array([], dtype="int")
        self.repeat_deg = np.array([], dtype="int")
        self.obs_i = np.array([], dtype="int")

        for i in range(len(self.messages.idx_list)):
            # add messages incoming to node i to self.inc_msgs
            self.inc_msgs = np.concatenate(
                (self.inc_msgs, self.messages.idx_list[i]), axis=0
            )
            num_neighbours = len(self.messages.idx_list[i])
            self.repeat_deg = np.append(self.repeat_deg, num_neighbours)
            for j in range(num_neighbours):
                # get the inverse of messages just added to self.inc_msgs
                k = self.messages.adj_list[i][j]
                self.obs_i = np.concatenate((self.obs_i, np.array([i])))
                jnd = np.where(np.array(self.messages.adj_list[k]) == i)[0]
                self.out_msgs = np.concatenate(
                    (self.out_msgs, self.messages.idx_list[k][jnd]), axis=0
                )

        self.reduce_idxs = np.delete(np.cumsum(self.repeat_deg), -1)
        self.reduce_idxs = np.concatenate((np.array([0]), self.reduce_idxs), axis=0)

        if verbose:
            print("Lists of neighbors created")

        # define Lambda tensors used in update
        self.Lambda0_tilde = np.copy(self.Lambda0.values[self.inc_msgs])
        self.Lambda1_tilde = np.copy(self.Lambda1.values[self.inc_msgs])
        if verbose:
            print("Copied lambdas matrices computed")

    def get_gamma(self, arr, reduce_idxs, repeat_deg):
        epsilon = 1e-20
        arr[arr == 0] = epsilon
        arr_2 = np.log(arr)
        arr_copy = np.copy(arr_2)
        arr_3 = np.add.reduceat(arr_2, reduce_idxs, axis=0)
        arr_4 = np.repeat(arr_3, repeat_deg, axis=0)
        arr_5 = np.exp(arr_4 - arr_copy)
        return arr_5

    def iterate(self):
        T = self.time
        old_msgs = np.copy(self.messages.values)
        msgs_tilde = old_msgs[self.inc_msgs]

        # calculate gamma matrices
        gamma0_hat = np.sum(msgs_tilde * self.Lambda0_tilde, axis=1, keepdims=1)
        gamma1_hat = np.sum(msgs_tilde * self.Lambda1_tilde, axis=1, keepdims=1)
        gamma0 = self.get_gamma(gamma0_hat, self.reduce_idxs, self.repeat_deg)
        gamma1 = self.get_gamma(gamma1_hat, self.reduce_idxs, self.repeat_deg)

        # calculate part one of update
        one_obs = (1 - self.delta) * np.reshape(
            self.observations[self.obs_i], (len(self.out_msgs), 1, T + 2)
        )
        # due to floating point errors accrued when using np.log and np.exp
        # the substraction can sometimes give an extemely small negative result.
        # Therefore a hard cutoff is implemented to bring these values to zero.
        one_main = np.clip(
            self.Lambda1_tilde * gamma1 - self.Lambda0_tilde * gamma0, 0, 1
        )
        one = np.transpose(one_obs * one_main, (0, 2, 1))[:, 1 : T + 1, :]

        # calculate part two of update
        two_obs = self.delta * self.observations[self.obs_i][:, 0]
        two_msgs = np.sum(msgs_tilde[:, :, 0], axis=1)
        two_main = self.get_gamma(two_msgs, self.reduce_idxs, self.repeat_deg)
        two = np.reshape(
            np.tile(np.reshape(two_obs * two_main, (len(self.out_msgs), 1)), T + 2),
            (len(self.out_msgs), 1, T + 2),
        )

        # calculate part three of update
        three_obs = (1 - self.delta) * np.reshape(
            self.observations[self.obs_i][:, T + 1], (len(self.out_msgs), 1)
        )
        gamma1_reshaped = np.reshape(gamma1[:, 0, T + 1], (len(self.out_msgs), 1))
        three_main = self.Lambda1_tilde[:, :, T + 1] * gamma1_reshaped
        three = np.reshape(three_obs * three_main, (len(self.out_msgs), 1, T + 2))

        # update the message values
        update_one = np.concatenate(
            (
                np.zeros((len(self.out_msgs), 1, T + 2)),
                one,
                np.zeros((len(self.out_msgs), 1, T + 2)),
            ),
            axis=1,
        )
        update_two = np.concatenate(
            (two, np.zeros((len(self.out_msgs), T + 1, T + 2))), axis=1
        )
        update_three = np.concatenate(
            (np.zeros((len(self.out_msgs), T + 1, T + 2)), three), axis=1
        )
        new_msgs = update_one + update_two + update_three
        norm = np.reshape(np.sum(new_msgs, axis=(1, 2)), (len(self.out_msgs), 1, 1))
        self.messages.values[self.out_msgs] = new_msgs / norm
        difference = np.abs(old_msgs - self.messages.values).max()

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
            indices = [np.random.randint(0, N) for _ in range(c - 1)]
            inc_msgs = np.array([old_msgs.values[idx] for idx in indices])
            inc_lambda0 = np.array([self.Lambda0.values[idx] for idx in indices])
            inc_lambda1 = np.array([self.Lambda1.values[idx] for idx in indices])
            gamma0_ki = np.reshape(
                np.prod(np.sum(inc_lambda0 * inc_msgs, axis=1), axis=0), (1, T + 2)
            )
            gamma1_ki = np.reshape(
                np.prod(np.sum(inc_lambda1 * inc_msgs, axis=1), axis=0), (1, T + 2)
            )
            self.messages.values[i] = np.transpose(
                (
                    (1 - self.delta)
                    * np.reshape(np.ones(T + 2), (1, T + 2))
                    * (inc_lambda1[0] * gamma1_ki - inc_lambda0[0] * gamma0_ki)
                )
            )
            self.messages.values[i][0] = self.delta * np.prod(
                np.sum(inc_msgs[:, :, 0], axis=1), axis=0
            )
            self.messages.values[i][T + 1] = np.transpose(
                (1 - self.delta) * inc_lambda1[0][:, T + 1] * gamma1_ki[0][T + 1]
            )
            norm = self.messages.values[i].sum()  # normalize the messages
            self.messages.values[i] = self.messages.values[i] / norm

        difference = np.abs(old_msgs.values - self.messages.values).max()

        return difference

    def update(self, maxit=100, tol=1e-6, print_iter=None):
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
        while i < maxit and error > tol:
            error = self.iterate()
            i += 1
            if print_iter != None:
                print_iter(error, i)
        return i, error

    def marginals(self):
        """Computes the array of the BP marginals for each node

        Returns:
            marginals (np.array): Array of the BP marginals, of shape N x (T+2)
        """

        marginals = []
        # we have one marginal (Tx1 vector) for each node.
        for n in range(self.size):  # loop through all nodes
            inc_indices, out_indices = self.messages.get_all_indices(n)
            inc_msg = self.messages.values[
                inc_indices[0]
            ]  # b_i(t_i) is the same regardless of which non directed edge (ij), j \in\partial i we pick, so long as we sum over j.
            out_msg = self.messages.values[out_indices[0]]
            marg = np.sum(
                inc_msg * np.transpose(out_msg), axis=0
            )  # transpose outgoing message so index to sum over after broadcasting is 0.
            marginals.append(marg / marg.sum())
        return np.asarray(marginals)

    def loglikelihood(self):
        """Computes the LogLikelihood from the BP messages

        Returns:
            logL (float): LogLikelihood
        """
        T = self.time

        log_zi = 0.0
        dummy_array = np.zeros((1, T + 2))
        for i in range(self.size):
            inc_indices, out_indices = self.messages.get_all_indices(i)
            inc_msgs = self.messages.get_neigh_i(i)
            inc_lambda0 = self.Lambda0.get_neigh_i(i)
            inc_lambda1 = self.Lambda1.get_neigh_i(i)

            gamma0_ki = np.reshape(
                np.prod(np.sum(inc_lambda0 * inc_msgs, axis=1), axis=0), (1, T + 2)
            )
            gamma1_ki = np.reshape(
                np.prod(np.sum(inc_lambda1 * inc_msgs, axis=1), axis=0), (1, T + 2)
            )
            dummy_array = np.transpose(
                (
                    (1 - self.delta)
                    * np.reshape(self.observations[i], (1, T + 2))
                    * (gamma1_ki - gamma0_ki)
                )
            )
            dummy_array[0] = (
                self.delta
                * self.observations[i][0]
                * np.prod(np.sum(inc_msgs[:, :, 0], axis=1), axis=0)
            )
            dummy_array[T + 1] = np.transpose(
                (1 - self.delta) * self.observations[i][T + 1] * gamma1_ki[0][T + 1]
            )
            log_zi = log_zi + np.log(dummy_array.sum())

        log_zij = 0.0
        # we have one marginal (Tx1 vector) for each node.
        for n in range(self.size):  # loop through all nodes
            inc_indices, out_indices = self.messages.get_all_indices(n)
            for j in np.arange(0, len(out_indices)):
                inc_msg = self.messages.values[
                    inc_indices[j]
                ]  # b_i(t_i) is the same regardless of which non directed edge (ij), j \in\partial i we pick, so long as we sum over j.
                out_msg = self.messages.values[out_indices[j]]
                marg = np.sum(
                    inc_msg * np.transpose(out_msg), axis=0
                )  # transpose outgoing message so index to sum over after broadcasting is 0.
                log_zij = log_zij + np.log(marg.sum())
        return log_zi - 0.5 * log_zij

    def reset_obs(self, obs):
        """Resets the observations, starting from the obs list

        Args:
            obs (list): List of the observations, each given by a list (i, 0/1, t), where 0 corresponds to S and 1 to I

        """
        self.observations = np.ones(
            (self.size, self.time + 2)
        )  # creating the mask for observations
        for o in obs:
            if o[1] == 1:
                self.observations[o[0]][o[2] + 1 :] = 0
            if o[1] == 0:
                self.observations[o[0]][: o[2] + 1] = 0
