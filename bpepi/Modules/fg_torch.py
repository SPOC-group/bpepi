# import numpy as np
from bpepi.Modules.st_torch import *
import torch


def delete(arr: torch.Tensor, ind: int, dim: int) -> torch.Tensor:
    skip = [i for i in range(arr.size(dim)) if i != ind]
    indices = [slice(None) if i != dim else skip for i in range(arr.ndim)]
    return arr.__getitem__(indices)


class FactorGraph:
    """Class to update the BP messages for the SI model"""

    def __init__(
        self,
        N,
        T,
        contacts,
        obs,
        delta,
        mask=["SI"],
        mask_type="SI",
        verbose=False,
        device="cpu",
        dtype=torch.float,
    ):
        """Construction of the FactorGraph object, starting from contacts and observations

        Args:
            N (int): Number of nodes in the contact network
            T (int): Time at which the simulation stops
            contacts (list): List of all the contacts, each given by a list (i, j, t, lambda_ij(t) )
            obs (list): List of the observations, each given by a list (i, 0/1, t), where 0 corresponds to S and 1 to I
            delta (float): Probability for an individual to be a source
            mask (list): if it is equal to ["SI"], the function simulates an SI model, otherwise the i-th element of the
                list (between 0 and 1) represents the infectivity of the nodes at i timesteps after the infection
            mask_type (string): Type of inference model. If equal to "SIR", it means we are simulating a SIR model
                and inferring using the dSIR model
        """
        self.messages = SparseTensor(
            N, T, contacts, device=device, dtype=dtype)
        if verbose:
            print("Messages matrices created")

        self.size = N
        self.time = T
        self.delta = delta
        self.contacts = contacts

        self.device = device
        self.dtype = dtype

        self.Lambda0 = SparseTensor(
            Tensor_to_copy=self.messages, device=device, dtype=dtype
        )  # messages only depend on lambda through Lambda matrices.
        self.Lambda1 = SparseTensor(
            Tensor_to_copy=self.messages, device=device, dtype=dtype
        )

        if verbose:
            print("Lambdas matrices created")

        if mask == ["SI"]:
            compute_Lambdas(self.Lambda0, self.Lambda1, contacts)
        else:
            compute_Lambdas_dSIR(self.Lambda0, self.Lambda1, contacts, mask)

        if verbose:
            print("Lambdas matrices computed")

        # creating the mask for observations
        self.observations = torch.ones(
            (self.size, self.time + 2)
        ) 
        
        self.reset_obs(obs)
        if verbose:
            print("Observations array created")

        self.out_msgs = torch.tensor([], dtype=torch.int, device=device)
        self.inc_msgs = torch.tensor([], dtype=torch.int, device=device)
        self.repeat_deg = []
        self.obs_i = torch.tensor([], dtype=torch.int, device=device)

        for i in range(len(self.messages.idx_list)):
            # add messages incoming to node i to self.inc_msgs
            self.inc_msgs = torch.concat(
                (self.inc_msgs, self.messages.idx_list[i]), axis=0
            )
            num_neighbours = len(self.messages.idx_list[i])
            self.repeat_deg.append(num_neighbours)
            for j in range(num_neighbours):
                # get the inverse of messages just added to self.inc_msgs
                k = self.messages.adj_list[i][j]
                self.obs_i = torch.concat((self.obs_i, torch.tensor([i])))
                jnd = torch.where(torch.tensor(
                    self.messages.adj_list[k]) == i)[0]
                self.out_msgs = torch.concat(
                    (self.out_msgs, self.messages.idx_list[k][jnd]), axis=0
                )
        self.repeat_deg = torch.tensor(
            self.repeat_deg, dtype=torch.int, device=device)
        self.reduce_idxs = torch.arange(
            len(self.repeat_deg), device=device
        ).repeat_interleave(self.repeat_deg)

        if verbose:
            print("Lists of neighbors created")

        # define Lambda tensors used in update
        self.Lambda0_tilde = torch.clone(self.Lambda0.values[self.inc_msgs])
        self.Lambda1_tilde = torch.clone(self.Lambda1.values[self.inc_msgs])
        if verbose:
            print("Copied lambdas matrices computed")

    def get_gamma(self, arr, reduce_idxs, repeat_deg):
        """Function to compute ratios between gamma functions in iterate()

        Args:
            arr (array): initial array
            reduce_idxs (array): list of indexes necessary in the function reduceat()
            repeat_deg (array):list of degrees necessary in the function repeat()

        Returns:
            arr_5 (array): final array
        """
        epsilon = 1e-20
        arr[arr == 0] = epsilon
        arr_2 = torch.log(arr)
        arr_copy = torch.clone(arr_2)
        size = list(arr.size())
        size[0] = self.size
        arr_3 = torch.zeros(size, device=self.device, dtype=self.dtype)
        arr_3.index_add_(0, reduce_idxs, arr_2)
        arr_4 = torch.repeat_interleave(arr_3, repeat_deg, dim=0)
        arr_5 = torch.exp(arr_4 - arr_copy)
        return arr_5

    def iterate(self, damp):
        """Single iteration of the Belief Propagation algorithm

        Returns:
            difference (float): Maximum difference between the messages at two consecutive iterations
        """
        T = self.time
        old_msgs = torch.clone(self.messages.values)
        msgs_tilde = old_msgs[self.inc_msgs]

        # calculate gamma matrices
        gamma0_hat = torch.sum(
            msgs_tilde * self.Lambda0_tilde, dim=1, keepdim=True)
        gamma1_hat = torch.sum(
            msgs_tilde * self.Lambda1_tilde, dim=1, keepdim=True)

        gamma0 = self.get_gamma(gamma0_hat, self.reduce_idxs, self.repeat_deg)
        gamma1 = self.get_gamma(gamma1_hat, self.reduce_idxs, self.repeat_deg)

        # calculate part one of update
        one_obs = (1 - self.delta) * torch.reshape(
            self.observations[self.obs_i], (len(self.out_msgs), 1, T + 2)
        )
        # due to floating point errors accrued when using torch.log and torch.exp
        # the substraction can sometimes give an extemely small negative result.
        # Therefore a hard cutoff is implemented to bring these values to zero.
        one_main = torch.clip(
            self.Lambda1_tilde * gamma1 - self.Lambda0_tilde * gamma0, 0, 1
        )
        one = torch.permute(one_obs * one_main, (0, 2, 1))[:, 1: T + 1, :]

        # calculate part two of update
        two_obs = self.delta * self.observations[self.obs_i][:, 0]
        two_msgs = torch.sum(msgs_tilde[:, :, 0], axis=1)
        two_main = self.get_gamma(two_msgs, self.reduce_idxs, self.repeat_deg)
        two = torch.reshape(
            torch.tile(
                torch.reshape(two_obs * two_main,
                              (len(self.out_msgs), 1)), (1, T + 2)
            ),
            (len(self.out_msgs), 1, T + 2),
        )

        # calculate part three of update
        three_obs = (1 - self.delta) * torch.reshape(
            self.observations[self.obs_i][:, T + 1], (len(self.out_msgs), 1)
        )
        gamma1_reshaped = torch.reshape(
            gamma1[:, 0, T + 1], (len(self.out_msgs), 1))
        three_main = self.Lambda1_tilde[:, :, T + 1] * gamma1_reshaped
        three = torch.reshape(three_obs * three_main,
                              (len(self.out_msgs), 1, T + 2))

        # update the message values
        update_one = torch.concat(
            (
                torch.zeros(
                    (len(self.out_msgs), 1, T + 2), device=self.device, dtype=self.dtype
                ),
                one,
                torch.zeros(
                    (len(self.out_msgs), 1, T + 2), device=self.device, dtype=self.dtype
                ),
            ),
            axis=1,
        )
        update_two = torch.concat(
            (
                two,
                torch.zeros(
                    (len(self.out_msgs), T + 1, T + 2),
                    device=self.device,
                    dtype=self.dtype,
                ),
            ),
            axis=1,
        )
        update_three = torch.concat(
            (
                torch.zeros(
                    (len(self.out_msgs), T + 1, T + 2),
                    device=self.device,
                    dtype=self.dtype,
                ),
                three,
            ),
            axis=1,
        )
        new_msgs = update_one + update_two + update_three
        norm = torch.reshape(
            torch.sum(new_msgs, axis=(1, 2)), (len(self.out_msgs), 1, 1)
        )
        norm_msgs = new_msgs / norm

        torch.nan_to_num(norm_msgs, nan=1./(self.time+2), out=norm_msgs)
        self.messages.values[self.out_msgs] = (1 - damp) * norm_msgs + damp * old_msgs[
            self.out_msgs
        ]  # Add dumping
        err_array = torch.abs(old_msgs - self.messages.values)

        return err_array.max().numpy(), err_array.mean().numpy()

    def pop_dyn_RRG(self, c=3):
        """Single iteration of the Population dynamics algorithm for a d-RRG

        Args:
            c (int): degree of the RRG

        Returns:
            difference (float): Maximum difference between the messages at two consecutive iterations
        """
        T = self.time
        N = self.size
        old_msgs = torch.clone(self.messages.values)
        for i in range(N):
            indices = [torch.random.randint(0, N) for _ in range(c - 1)]
            inc_msgs = torch.tensor(
                [old_msgs[idx] for idx in indices], device=self.device, dtype=self.dtype
            )
            inc_lambda0 = torch.tensor(
                [self.Lambda0.values[idx] for idx in indices],
                device=self.device,
                dtype=self.dtype,
            )
            inc_lambda1 = torch.tensor(
                [self.Lambda1.values[idx] for idx in indices],
                device=self.device,
                dtype=self.dtype,
            )
            gamma0_ki = torch.reshape(
                torch.prod(torch.sum(inc_lambda0 * inc_msgs, axis=1), axis=0),
                (1, T + 2),
            )
            gamma1_ki = torch.reshape(
                torch.prod(torch.sum(inc_lambda1 * inc_msgs, axis=1), axis=0),
                (1, T + 2),
            )
            self.messages.values[i] = torch.transpose(
                (
                    (1 - self.delta)
                    * torch.reshape(torch.ones(T + 2), (1, T + 2))
                    * (inc_lambda1[0] * gamma1_ki - inc_lambda0[0] * gamma0_ki)
                )
            )
            self.messages.values[i][0] = self.delta * torch.prod(
                torch.sum(inc_msgs[:, :, 0], axis=1), axis=0
            )
            self.messages.values[i][T + 1] = torch.transpose(
                (1 - self.delta) *
                inc_lambda1[0][:, T + 1] * gamma1_ki[0][T + 1]
            )
            norm = self.messages.values[i].sum()  # normalize the messages
            self.messages.values[i] = self.messages.values[i] / norm

        difference = torch.abs(old_msgs - self.messages.values).max()

        return difference

    def update(self, maxit=100, tol=1e-6, damp=0.0, print_iter=None):
        """Multiple iterations of the BP algorithm through the method iterate()

        Args:
            maxit (int, optional): Maximum number of iterations of BP. Defaults to 100.
            tol (double, optional): Tolerance threshold for the difference between consecutive. Defaults to 1e-6.

        Returns:
            i (int): Iteration at which the algorithm stops
            error (float): Error on the messages at the end of the iterations
        """
        i = 0
        error_mean = 1
        while i < maxit and error_mean > tol:
            error_max, error_mean = self.iterate(damp)
            i += 1
            if print_iter != None:
                print_iter([error_max, error_mean], i)
        return i, [error_max, error_mean]

    def marginals(self):
        """Computes the array of the BP marginals for each node

        Returns:
            marginals (torch.tensor): Array of the BP marginals, of shape N x (T+2)
        """

        marginals = torch.zeros(
            (self.size, self.time + 2), device=self.device, dtype=self.dtype
        )
        # we have one marginal (Tx1 vector) for each node.
        for n in range(self.size):  # loop through all nodes
            inc_indices, out_indices = self.messages.get_all_indices(n)
            inc_msg = self.messages.values[
                inc_indices[0]
            ]  # b_i(t_i) is the same regardless of which non directed edge (ij), j \in\partial i we pick, so long as we sum over j.
            out_msg = self.messages.values[out_indices[0]]
            marg = torch.sum(
                inc_msg * torch.t(out_msg), dim=0
            )  # transpose outgoing message so index to sum over after broadcasting is 0.
            # DEBUG CHECK
            # if (marg.sum()==0):
            #    print(f"INC{inc_msg}")
            #    print(f"OUT{torch.transpose(out_msg)}")
            marginals[n, :] = marg / marg.sum()
        return marginals.numpy()
    
    def get_messages(self):
        """Computes the array of the BP marginals for each node

        Returns:
            marginals (np.array): Array of the BP marginals, of shape N x (T+2)
        """

        #mess = SparseTensor(
        #    Tensor_to_copy=self.messages
        #)
        #mess.values = np.copy(self.messages.values)
        return self.messages.values

    def loglikelihood(self):
        """Computes the LogLikelihood from the BP messages

        Returns:
            logL (float): LogLikelihood
        """
        T = self.time

        log_zi = 0.0
        dummy_array = torch.zeros((1, T + 2))
        for i in range(self.size):
            inc_indices, out_indices = self.messages.get_all_indices(i)
            inc_msgs = self.messages.get_neigh_i(i)
            inc_lambda0 = self.Lambda0.get_neigh_i(i)
            inc_lambda1 = self.Lambda1.get_neigh_i(i)

            gamma0_ki = torch.reshape(
                torch.prod(torch.sum(inc_lambda0 * inc_msgs, axis=1), axis=0),
                (1, T + 2),
            )
            gamma1_ki = torch.reshape(
                torch.prod(torch.sum(inc_lambda1 * inc_msgs, axis=1), axis=0),
                (1, T + 2),
            )
            dummy_array = torch.t(
                (
                    (1 - self.delta)
                    * torch.reshape(self.observations[i], (1, T + 2))
                    * (gamma1_ki - gamma0_ki)
                )
            )
            dummy_array[0] = (
                self.delta
                * self.observations[i][0]
                * torch.prod(torch.sum(inc_msgs[:, :, 0], axis=1), axis=0)
            )
            dummy_array[T + 1] = torch.t(
                (1 - self.delta) *
                self.observations[i][T + 1] * gamma1_ki[0][T + 1]
            )
            log_zi = log_zi + torch.log(dummy_array.sum())

        log_zij = 0.0
        # we have one marginal (Tx1 vector) for each node.
        for n in range(self.size):  # loop through all nodes
            inc_indices, out_indices = self.messages.get_all_indices(n)
            for j in torch.arange(0, len(out_indices)):
                inc_msg = self.messages.values[
                    inc_indices[j]
                ]  # b_i(t_i) is the same regardless of which non directed edge (ij), j \in\partial i we pick, so long as we sum over j.
                out_msg = self.messages.values[out_indices[j]]
                marg = torch.sum(
                    inc_msg * torch.t(out_msg), axis=0
                )  # transpose outgoing message so index to sum over after broadcasting is 0.
                log_zij = log_zij + torch.log(marg.sum())
        return log_zi - 0.5 * log_zij

    def reset_obs(self, obs):
        """Resets the observations, starting from the obs list

        Args:
            obs (list): List of the observations, each given by a list (i, 0/1, t), where 0 corresponds to S and 1 to I

        """
        self.observations = torch.ones(
            (self.size, self.time + 2)
        )  # creating the mask for observations
        for o in obs:
            if o[1] == 0:
                self.observations[o[0]][: o[2] + 1] = 0
            else:
                self.observations[o[0]][o[2] + 1:] = 0
