import numpy as np
import random
import networkx

def simulate_one_detSIR(G, p_inf = 0.01, mask = ["SI"]):
    """Single iteration of the Population dynamics algorithm for a d-RRG

    Args:
        G (nx.graph): Graph representing the contact network
        p_inf (float): probability of being infected at time 0 (fraction of sources)
        mask (list): if it is equal to ["SI"], the function simulates an SI model, otherwise the i-th element of the
            list (between 0 and 1) represents the infectivity of the nodes at i timesteps after the infection

    Returns:
        status_nodes (list): array of shape (T+1) x N containing the state of all the nodes from time 0 to time T
    """
    N=G.number_of_nodes()
    # Generate the sources
    status_nodes = []
    s0 = []
    flag_s=0
    flag_m = 1
    if mask == ["SI"] : 
        mask = [1]
        flag_m = 0
    counter = [mask.copy() for _ in range(N)]
    coeff_lam = np.ones(N)
    for i in range(N):
        if np.random.rand() < p_inf : 
            s0.append(1)
            coeff_lam[i]=counter[i][0]
            if flag_m : counter[i].pop(0)
            flag_s =1
        else : s0.append(0)

    if (flag_s==0) : 
        s0[np.random.randint(0,N)]=1
        print("No sources... adding a single random source")
    status_nodes.append(np.array(s0))
    # Generate the epidemics
    st = np.copy(s0)
    while ((flag_m==1 and 1 in status_nodes[-1]) or (flag_m==0 and 0 in status_nodes[-1])):
        st_minus_1 = np.copy(st)
        coeff_minus_1 = coeff_lam.copy()
        for i in range(N):
            if st[i] == 1 :
                if not counter[i]: st[i]=2
                else :
                    coeff_lam[i]=counter[i][0]
                    if flag_m : counter[i].pop(0)
            elif st[i] == 0 :
                for j in nx.neighbors(G,i) :
                    if (st_minus_1[j]==1 and np.random.rand() < G.edges[j,i]['lambda']*coeff_minus_1[j] and st[i]==0): 
                        st[i]=1
                        coeff_lam[i]=counter[i][0]
                        if flag_m : counter[i].pop(0)
        status_nodes.append(np.copy(st))
    return np.array(status_nodes)


def generate_contacts(G, T, lambda_, p_edge=1):
    """Function to generate contacts between nodes, given the probability lambda_

    Args:
        G (nx.graph): Graph representing the contact network
        T (int): time at which the epidemic stops
        lambda_ (float): probability of infection, constant for all edges
        p_edge (float): probability of having a contact


    Returns:
        contacts (list): List of directed contacts, each of the form (i,j,t,lambda)
    """
    contacts = []
    for t in range(T):
        for e in G.edges():
            if random.random() <= p_edge:
                contacts.append((e[0], e[1], t, lambda_))
                contacts.append((e[1], e[0], t, lambda_))
    return contacts


def generate_M_obs(conf, M=0):
    """Function to generate M observations, given an epidemic simulation

    Args:
        conf (array): array of shape (T+1) x N contaning the states of all the nodes from time 0 to time T
        M (int): Number of observations. Defaults to 0.

    Returns:
        obs_sim (list): list of observations, each of the form (i,0/1,t) where 0/1 is a negative/positive test
        fI (float): always 1
    """
    obs_sim = []
    N = len(conf[0])
    T = len(conf)
    obs_list = random.sample(range(N), M)
    for i in obs_list:
        t_inf = np.nonzero(conf[:, i] == 1)[0]
        if len(t_inf) == 0:
            obs_temp = (i, 0, T - 1)
            obs_sim.append(obs_temp)
        else:
            obs_temp = (i, 1, t_inf[0])
            obs_sim.append((obs_temp))
            if t_inf[0] > 0:
                obs_temp = (i, 0, t_inf[0] - 1)
                obs_sim.append((obs_temp))
    obs_sim = sorted(obs_sim, key=lambda tup: tup[2])
    fI = 1
    return obs_sim, fI


def generate_snapshot_obs(conf, frac_obs=0.0):
    """Function to generate a snapshot observation, given an epidemic simulation

    Args:
        conf (array): array of shape (T+1) x N contaning the states of all the nodes from time 0 to time T
        frac_obs (float): fraction of nodes to observe. Defaults to 0.

    Returns:
        obs_sim (list): list of observations, each of the form (i,0/1,t) where 0/1 is a negative/positive test
        fI (float): fraction of infected nodes when taking the tests
        tS (int): random time at which the tests were taken
    """
    obs_sim = []
    N = len(conf[0])
    T = len(conf)
    tS = random.choice(range(T))
    fI = conf[tS].mean()
    obs_sim = [ (i, conf[tS,i], tS) for i in range(N) if np.random.random() < frac_obs]
    return obs_sim, fI, tS

# TO BE WRITTEN
#def generate_snapshot_obs_dSIR(conf, frac_obs=0.0):
#    """[summary]
#
#    Args:
#        conf ([type]): [description]
#        frac_obs (float, optional): [description]. Defaults to 0.1.
#
#    Returns:
#        [type]: [description]
#    """
#    obs_sim = []
#    N = len(conf[0])
#    T = len(conf)
#    tS = random.choice(range(T))
#    fI = conf[tS].mean()
#    for i in range(N):
#        if np.random.random() < frac_obs :
#            if conf[tS,i]!= 2 : obs_sim.append( (i, conf[tS,i], tS) )
#            else : obs_sim.append( (i, 1, tS-Delta) )
#    return obs_sim, fI, tS
#

def generate_obs(conf, frac_obs=0.0):
    """Function to generate a list of observations of a certain fraction of nodes, given an epidemic simulation

    Args:
        conf ([type]): [description]
        frac_obs (float): fraction of nodes to observe. Defaults to 0.

    Returns:
        obs_sim (list): list of observations, each of the form (i,0/1,t) where 0/1 is a negative/positive test
        fI (float): always 1
    """
    obs_sim = []
    N = conf.shape[1]
    T = conf.shape[0]
    for i in range(N):
        if np.random.random() < frac_obs:
            t_inf = np.nonzero(conf[:, i] == 1)[0]
            if len(t_inf) == 0:
                obs_temp = (i, 0, T - 1)
                obs_sim.append(obs_temp)
            else:
                obs_temp = (i, 1, t_inf[0])
                obs_sim.append((obs_temp))
                if t_inf[0] > 0:
                    obs_temp = (i, 0, t_inf[0] - 1)
                    obs_sim.append((obs_temp))
    obs_sim = sorted(obs_sim, key=lambda tup: tup[2])
    fI = 1
    return obs_sim, fI