import numpy as np
import random
import networkx as nx
import ndlib.models.epidemics as ep
import ndlib.models.ModelConfig as mc

def simulate_one_detSIR(G, s_type = "delta", S = 0.01, mask = ["SI"]):
    """Function to simulate an epidemic using the deterministic-SIR model

    Args:
        G (nx.graph): Graph representing the contact network
        s_type (string): either "delta" or "n_sources", indicates how to interpret the parameter S
        S (int/float): number of infected/probability of being infected at time 0 (number/fraction of sources)
        mask (list): if it is equal to ["SI"], the function simulates an SI model, otherwise the i-th element of the
            list (between 0 and 1) represents the infectivity of the nodes at i timesteps after the infection

    Returns:
        status_nodes (list): array of shape (T+1) x N containing the state of all the nodes from time 0 to time T
    """
    N=G.number_of_nodes()
    # Generate the sources
    status_nodes = []
    flag_s=0
    flag_m = 1
    if mask == ["SI"] : 
        mask = [1]
        flag_m = 0
    counter = [mask.copy() for _ in range(N)]
    coeff_lam = np.ones(N)
    if s_type == "delta":
        s0 = []
        for i in range(N):
            if np.random.rand() < S : 
                s0.append(1)
                coeff_lam[i]=counter[i][0]
                if flag_m : counter[i].pop(0)
                flag_s =1
            else : s0.append(0)        
    else:
        flag_s=1
        s0=np.zeros(N)
        source_list = random.sample(range(N), S)
        s0[source_list]=1
        for i in source_list:
            coeff_lam[i]=counter[i][0]
            if flag_m : counter[i].pop(0)

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

def generate_snapshot_obs(conf, o_type="rho", M=0.0, snap_time=-1):
    """Function to generate a snapshot observation, given an epidemic simulation

    Args:
        conf (array): array of shape (T+1) x N contaning the states of all the nodes from time 0 to time T
        o_type (string): either "rho" or "n_obs", indicates how to interpret the parameter M
        M (int/float): number of observed nodes/probability of being randomly observed 

    Returns:
        obs_sim (list): list of observations, each of the form (i,0/1,t) where 0/1 is a negative/positive test
        fS (float): fraction of susceptible nodes when taking the tests
        fI (float): fraction of infected nodes when taking the tests
        tS (int): random time at which the tests were taken
    """
    obs_sim = []
    N = len(conf[0])
    T = len(conf)
    if snap_time == -1: snap_time = random.choice(range(T))
    fS = np.count_nonzero(conf[snap_time] == 0)
    fI = np.count_nonzero(conf[snap_time] == 1)
    if o_type == "rho":
        obs_sim = [ (i, conf[snap_time,i], snap_time) for i in range(N) if np.random.random() < M]
    else: 
        obs_list = random.sample(range(N), M)
        obs_sim = [ (i, conf[snap_time,i], snap_time) for i in range(N) if i in obs_list]
    return obs_sim, fS, fI, snap_time

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

def generate_obs(conf, o_type="rho", M=0.0):
    """Function to generate a list of observations of a certain fraction of nodes, given an epidemic simulation

    Args:
        conf ([type]): [description]
        o_type (string): either "rho" or "n_obs", indicates how to interpret the parameter M
        M (int/float): number of observed nodes/probability of being randomly observed 

    Returns:
        obs_sim (list): list of observations, each of the form (i,0/1,t) where 0/1 is a negative/positive test
    """
    obs_sim = []
    N = conf.shape[1]
    T = conf.shape[0]
    if o_type == "n_obs":
        obs_list = random.sample(range(N), M)
    for i in range(N):
        if ((o_type == "rho") and (np.random.random() < M)) or ((o_type == "n_obs") and (i in obs_list)):
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
    return obs_sim

def simulate_one_SIR(G, s_type = "delta", S = 0.01, mu=0.):
    """Function to simulate an epidemic using the SIR model

    Args:
        G (nx.graph): Graph representing the contact network
        s_type (string): either "delta" or "n_sources", indicates how to interpret the parameter S
        S (int/float): number of infected/probability of being infected at time 0 (number/fraction of sources)
     
    Returns:
        status_nodes (list): array of shape (T+1) x N containing the state of all the nodes from time 0 to time T
    """
    N = G.number_of_nodes()
    for j in nx.neighbors(G,0):
        lamb = G.edges[j,0]['lambda']
    model = ep.SIRModel(G)
    cfg = mc.Configuration()
    cfg.add_model_parameter("beta", lamb)  # infection rate
    cfg.add_model_parameter('gamma', mu) # recovery rate
    infected_nodes = []
    for i in range(N) :
        if np.random.rand() < S : 
            infected_nodes.append(i)
    cfg.add_model_initial_configuration("Infected", infected_nodes)
    model.set_initial_status(cfg)
    status_nodes = []
    status_nodes_t = np.zeros(N, dtype=int)
    t = 0
    while sum(status_nodes_t == 1) > 1:
        iteration = model.iteration(node_status=True)
        for node_i in iteration["status"].keys():
            status_nodes_t[node_i] = iteration["status"][node_i]
        status_nodes.append(status_nodes_t.copy())
        t = t + 1

    return np.array(status_nodes)

def generate_obs_SIR(conf, o_type="rho", M=0.0):
    """Function to generate a list of observations of a certain fraction of nodes, given an epidemic simulation

    Args:
        conf ([type]): [description]
        o_type (string): either "rho" or "n_obs", indicates how to interpret the parameter M
        M (int/float): number of observed nodes/probability of being randomly observed 

    Returns:
        obs_sim (list): list of observations, each of the form (i,0/1,t) where 0/1 is a negative/positive test
    """
    obs_sim = []
    N = conf.shape[1]
    T = conf.shape[0]
    if o_type == "n_obs":
        obs_list = random.sample(range(N), M)
    for i in range(N):
        if ((o_type == "rho") and (np.random.random() < M)) or ((o_type == "n_obs") and (i in obs_list)):
            t_inf = np.nonzero(conf[:, i] == 1)[0]
            t_rec = np.nonzero(conf[:, i] == 2)[0]
            if len(t_inf) == 0:
                obs_temp = (i, 0, T - 1)
                obs_sim.append(obs_temp)
            else:
                obs_temp = (i, 1, t_inf[0])
                obs_sim.append((obs_temp))
                if t_inf[0] > 0:
                    obs_temp = (i, 0, t_inf[0] - 1)
                    obs_sim.append((obs_temp))
                if len(t_rec) == 0:
                    obs_temp = (i, 1, T - 1)
                    obs_sim.append(obs_temp)
                else:
                    obs_temp = (i, 2, t_rec[0])
                    obs_sim.append((obs_temp))
                    obs_temp = (i, 1, t_rec[0] - 1)
                    obs_sim.append((obs_temp))
    obs_sim = sorted(obs_sim, key=lambda tup: tup[2])
    return obs_sim