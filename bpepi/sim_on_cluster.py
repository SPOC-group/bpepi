import pandas as pd
import ndlib

# import random
import numpy as np
import networkx as nx
import random

from fg import FactorGraph
from st import SparseTensor
import sys, os
import time

import lzma
import pickle
from pathlib import Path
import argparse
import warnings


path_script = Path(sys.argv[0]).parent.absolute()
sys.path.append(os.fspath(path_script.parent / "src"))
print(path_script)

# sys.path.append('../../')
''''''''''''''''''''''''''''''''''''

import ndlib.models.epidemics as ep
import ndlib.models.ModelConfig as mc

def generate_one_conf(g, lamb=0.05, T=10, percentage_infected=0.01):
    model = ep.SIModel(g)
    cfg = mc.Configuration()
    cfg.add_model_parameter('beta', lamb) # infection rate
    #cfg.add_model_parameter('gamma', 0.0) # recovery rate
    cfg.add_model_parameter("percentage_infected", percentage_infected)
    model.set_initial_status(cfg)
    iterations = model.iteration_bunch(T, node_status=True)
    status_nodes = np.zeros((T,N))
    for t in range(0,T):
        for node_i in iterations[t]["status"].keys():
            status_nodes[t:T, node_i] = iterations[t]["status"][node_i]
    return np.array(status_nodes)
    
def generate_contacts(G, t_limit, lambda_, p_edge=1, seed=1):
    contacts = []
    random.seed(seed)
    for t in range(t_limit):
        for e in G.edges():
            if random.random() <= p_edge:
                contacts.append((e[0], e[1], t, lambda_))
                contacts.append((e[1], e[0], t, lambda_))
    #contacts = np.array(contacts, dtype=[("t","f8"), ("i","f8"), ("j","f8"), ("lam", "f8")])
    #contacts.sort(axis=0, order=("tù","i","j"))
    return contacts

def generate_M_obs(conf, M=0):
    """[summary]

    Args:
        conf ([type]): [description]
        frac_obs (float, optional): [description]. Defaults to 0.1.

    Returns:
        [type]: [description]
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
    return obs_sim

def generate_obs(conf, frac_obs=0.):
    """[summary]

    Args:
        conf ([type]): [description]
        frac_obs (float, optional): [description]. Defaults to 0.1.

    Returns:
        [type]: [description]
    """
    obs_sim = []
    N = conf.shape[1]
    T = conf.shape[0]
    for i in range(N):
        if np.random.random() < frac_obs: 
            t_inf = np.nonzero(conf[:, i] == 1)[0]
            if len(t_inf) == 0:
                obs_temp = (i, 0, T-1)
                obs_sim.append(obs_temp)
            else:
                obs_temp = (i, 1, t_inf[0])
                obs_sim.append((obs_temp))
                if t_inf[0] > 0:
                    obs_temp = (i, 0, t_inf[0] - 1)
                    obs_sim.append((obs_temp))
    obs_sim = sorted(obs_sim, key=lambda tup: tup[2])
    return obs_sim

def iter_print(t, e, f):
    print(f"", end="\r", flush=True)


def create_data_obs(flag_sources, flag_obs, n_sim, N, d, lam,n_iter, pseed):
    data_obs = {}
    data_obs["N"] = N
    data_obs["d"] = d
    data_obs["lam"] = lam
    data_obs["n_sim"] = n_sim
    data_obs["n_iter"] = n_iter
    data_obs["pseed"] = pseed
    if (flag_obs) : data_obs["rho"] = []
    else : data_obs["# obs"] = []
    if (flag_sources) : data_obs["delta"] = []
    else : data_obs["# sources"] = []
    data_obs["init"] = []
    data_obs["ov0"] = []  # overlap with the ground truth at t=0, n_T x n_M x n_sim
    data_obs["ov0_rnd"] = [] # overlap with the ground truth at t=0, n_T x n_M x n_sim
    data_obs["mov0"] = []  # mean overlap at t=0, n_T x n_M x n_sim
    data_obs["mov0_rnd"] = [] # overlap with the ground truth at t=0, n_T x n_M x n_sim
    data_obs["se"] = []
    data_obs["mse"] = []
    data_obs["nI"] = []
    data_obs["e"] = []
    data_obs["logL"] = []
    data_obs["marginal0"] = []
    data_obs["sim"] = []
    data_obs["T_table"] = []
    #data_obs["obs_table"] = []#np.zeros((n_M, n_sim, N))

    return data_obs


def fill_data_obs(
    data_obs, f, status_nodes, T, it, e, init, M, S, sim, flag_sources, flag_obs
):
    Bs = f.marginals()
    Ms0 = Bs[:,0]
    Ss = status_nodes
    ti_str = ti_star(Ss)

    data_obs["T_table"].append(T)
    #data_obs["obs_table"].append( obs_table)
    if (flag_obs) : data_obs["rho"].append( M )
    else : data_obs["# obs"].append( M )
    if (flag_sources) : data_obs["delta"].append( S )
    else : data_obs["# sources"].append( S )
    data_obs["sim"].append( sim )

    data_obs["init"].append( init )
    data_obs["ov0"].append( E_overlap(Ss[0], (Ms0 > 0.5)))
    data_obs["ov0_rnd"].append( E_overlap_rnd(Ss[0], Ms0))
    data_obs["mov0"].append( M_overlap(Ms0))
    data_obs["mov0_rnd"].append( M_overlap_rnd(Ms0))
    ti_inf = ti_inferred(Bs)
    data_obs["se"].append( E_SE(ti_str, ti_inf))
    data_obs["mse"].append( MSE(Bs, ti_inf))
    data_obs["nI"].append( it)
    data_obs["e"].append( e)
    data_obs["logL"].append( f.loglikelihood())
    data_obs["marginal0"].append( Ms0)


def create_data_obs_it(flag_sources, flag_obs, n_sim, N, d, lam,n_iter, pseed):
    data_obs_it = {}

    data_obs_it["N"] = N
    data_obs_it["d"] = d
    data_obs_it["lam"] = lam
    data_obs_it["n_sim"] = n_sim
    data_obs_it["n_iter"] = n_iter
    data_obs_it["pseed"] = pseed
    data_obs_it["init"] = []
    data_obs_it["sim"] = []
    data_obs_it["converged"] = []
    if (flag_obs) : data_obs_it["rho"] = []
    else : data_obs_it["# obs"] = []
    if (flag_sources) : data_obs_it["delta"] = []
    else : data_obs_it["# sources"] = []
    data_obs_it["iter"] = []
    data_obs_it["ov0"] = []  # overlap with the ground truth at t=0, n_T x n_M x n_sim
    data_obs_it["ov0_rnd"] = []  # overlap with the ground truth at t=0, n_T x n_M x n_sim
    data_obs_it["mov0"] = []  # mean overlap at t=0, n_T x n_M x n_sim
    data_obs_it["mov0_rnd"] = []  # overlap with the ground truth at t=0, n_T x n_M x n_sim
    data_obs_it["e"] = []
    data_obs_it["logL"] = []

    return data_obs_it


def fill_data_obs_it(data_obs_it, f, status_nodes, T, it, e, init, M, S, tol, n_iter, n_it_print, sim, flag_sources, flag_obs):
    Bs = f.marginals()
    Ms0 = Bs[:,0]
    Ss = status_nodes

    data_obs_it["init"].append(init)
    data_obs_it["sim"].append(sim)
    if (flag_obs) : data_obs_it["rho"].append( M )
    else : data_obs_it["# obs"].append( M )
    if (flag_sources) : data_obs_it["delta"].append( S )
    else : data_obs_it["# sources"].append( S )
    data_obs_it["iter"].append(it)
    data_obs_it["ov0"].append( E_overlap(Ss[0], (Ms0 > 0.5)) )
    data_obs_it["ov0_rnd"].append( E_overlap_rnd(Ss[0], Ms0) )
    data_obs_it["mov0"].append( M_overlap(Ms0) )
    data_obs_it["mov0_rnd"].append( M_overlap_rnd(Ms0) )
    data_obs_it["e"].append( e )
    if (e < tol): data_obs_it["converged"].extend(["yes"]*n_it_print)
    elif (it == n_iter): data_obs_it["converged"].extend(["no"]*n_it_print)
    data_obs_it["logL"].append( f.loglikelihood() )

def sim_and_fill(f, data_obs, data_obs_it, list_obs, n_iter, tol, print_it, status_nodes, T, init, M, S, iter_space, sim, flag_sources, flag_obs):
    for it in range(n_iter):
        e0 = f.iterate()
        if e0 < tol:
            break
    if (e0 > tol) : warnings.warn("Warning... Initialization is not converging")
    f.reset_obs(list_obs)

    e = np.nan
    n_it_print = 1
    if (print_it) : fill_data_obs_it(data_obs_it, f, status_nodes, T, 0, e, init, M, S, tol, n_iter, n_it_print, sim, flag_sources, flag_obs )
    for it in range(n_iter):
        e = f.iterate()
        if (print_it and (((it+1)%iter_space ==0) or (e < tol or (it+1) == n_iter)) ):
            n_it_print = n_it_print + 1
            fill_data_obs_it(data_obs_it, f, status_nodes, T, it+1, e, init, M, S, tol, n_iter, n_it_print, sim, flag_sources, flag_obs )
        if e < tol:
            break
    fill_data_obs( data_obs,
        f,
        status_nodes,
        T,
        it + 1,
        e,
        init,
        M,
        S,
        sim,
        flag_sources, 
        flag_obs
    )   

def generate_one_conf_ns_allI(g, lamb, n_sources):
    """[summary]

    Args:
        g ([type]): [description]
        lamb (float, optional): [description]. Defaults to 0.05.
        T (int, optional): [description]. Defaults to 10.
        percentage_infected (float, optional): [description]. Defaults to 0.01.

    Returns:
        [type]: [description]
    """
    N = g.number_of_nodes()
    model = ep.SIModel(g)
    cfg = mc.Configuration()
    cfg.add_model_parameter("beta", lamb)  # infection rate
    # cfg.add_model_parameter('gamma', 0.0) # recovery rate
    infected_nodes = random.sample(range(N), n_sources)
    cfg.add_model_initial_configuration("Infected", infected_nodes)
    model.set_initial_status(cfg)
    status_nodes = []
    status_nodes_t = np.zeros(N, dtype=int)
    t = 0
    while sum(status_nodes_t) < N:
        iteration = model.iteration(node_status=True)
        for node_i in iteration["status"].keys():
            status_nodes_t[node_i] = iteration["status"][node_i]
        status_nodes.append(status_nodes_t.copy())
        t = t + 1

    return np.array(status_nodes)


def generate_one_conf_delta_allI(g, lamb, delta):
    """[summary]

    Args:
        g ([type]): [description]
        lamb (float, optional): [description]. Defaults to 0.05.
        T (int, optional): [description]. Defaults to 10.
        percentage_infected (float, optional): [description]. Defaults to 0.01.

    Returns:
        [type]: [description]
    """
    N = g.number_of_nodes()
    model = ep.SIModel(g)
    cfg = mc.Configuration()
    cfg.add_model_parameter("beta", lamb)  # infection rate
    # cfg.add_model_parameter('gamma', 0.0) # recovery rate
    infected_nodes = []
    while (len(infected_nodes) == 0):
        for i in range(N):
            if np.random.random() < delta: infected_nodes.append(i)
    cfg.add_model_initial_configuration("Infected", infected_nodes)
    model.set_initial_status(cfg)
    status_nodes = []
    status_nodes_t = np.zeros(N, dtype=int)
    t = 0
    while sum(status_nodes_t) < N:
        iteration = model.iteration(node_status=True)
        for node_i in iteration["status"].keys():
            status_nodes_t[node_i] = iteration["status"][node_i]
        status_nodes.append(status_nodes_t.copy())
        t = t + 1

    return np.array(status_nodes)
'''
def marginals(f, T):
    """COmpute marginals of each node in the state S, I, R

    Args:
        f (sib.FactorGraph): Factor Graph of sib, iterated
        T ([type]): Total time of epidemic

    Returns:
        [numpy tensor]: Tensor of dimension (T, N, 3) containing the marginals probabilites to be SIR for each node at each time.
    """
    N = len(f.nodes)
    M = np.zeros((T, N, 3))
    for n in f.nodes:
        m = np.array(n.marginal())
        M[:, n.index] = m
    return M


def beliefs_ti(f, T):
    """COmpute marginals of each node in the state S, I, R

    Args:
        f (sib.FactorGraph): Factor Graph of sib, iterated
        T ([type]): Total time of epidemic

    Returns:
        [numpy tensor]: Tensor of dimension (T, N, 3) containing the marginals probabilites to be SIR for each node at each time.
    """
    N = len(f.nodes)
    B = np.zeros((N, T + 2))
    for n in f.nodes:
        b = np.array(n.bt)
        B[n.index] = b
    return B
'''

def ti_star(S):
    """[summary]

    Args:
        M ([type]): [description]

    Returns:
        [type]: [description]
    """
    N=len(S[0])
    T=len(S)
    ti = np.zeros(N)
    for i in range(N):
        t_inf = np.nonzero(S[:,i] == 1)[0]
        if len(t_inf) == 0: ti[i] =  T
        else: ti[i] = t_inf[0] -1
    return ti

def ti_inferred(B):
    """[summary]

    Args:
        M ([type]): [description]

    Returns:
        [type]: [description]
    """
    # T = len(B[0])
    return np.array(
        [
            np.array([(t-1) * bt for t, bt in enumerate(b)]).sum()
            for b in B
        ]
    )
    # return np.array([ np.array([ t*bt for t,bt in enumerate(b) if t != T ]).sum() for i,b in enumerate(B)])


def E_overlap(conf1, conf2, astype=bool, axis=(-1)):
    """[summary]

    Args:
        conf1 ([type]): [description]
        conf2 ([type]): [description]
        astype ([type], optional): [description]. Defaults to bool.

    Returns:
        [type]: [description]
    """
    ov_i = (conf1.astype(astype) == conf2.astype(astype)).mean(axis=axis)
    return ov_i


def M_overlap(M, axis=(-1)):
    """[summary]

    Args:
        M ([type]): [description]

    Returns:
        [type]: [description]
    """
    return np.maximum(M, 1-M).mean(axis=axis)

def E_overlap_rnd(conf, M, axis=-1):
    """
    [summary]

    Args:
        conf ([type]): [description]
    Returns:
        [type]: [description]
    """
    if (M.mean(axis=axis) > (1-M).mean(axis=axis)) : o = conf.mean(axis=axis)
    else : o = 1 - conf.mean(axis=axis)
    return o

def M_overlap_rnd(M, axis=-1):
    """
    [summary]

    Args:
        conf ([type]): [description]
    Returns:
        [type]: [description]
    """
    return np.maximum(M.mean(axis=axis), (1-M).mean(axis=axis))


def fraction_of_detected_I(conf1, conf2, list_obs):
    """[summary]

    Args:
        conf1 ([type]): [description]
        conf2 ([type]): [description]
        astype ([type], optional): [description]. Defaults to bool.

    Returns:
        [type]: [description]
    """
    di = 0.0
    nI = 0
    list_i = [l[0] for l in list_obs]
    for i, conf_i in enumerate(conf1):
        if (not (i in list_i)) and conf_i == 1:
            if conf2[i] == 1:
                di = di + 1
            nI = nI + 1
    if nI:
        f = di / nI
    else:
        f = float("nan")
    return f



def E_SE(ti, ti_inferred):
    """[summary]

    Args:
        conf1 ([type]): [description]
        conf2 ([type]): [description]
        astype ([type], optional): [description]. Defaults to bool.

    Returns:
        [type]: [description]
    """
    e_se = np.array([(t - ti_inferred[i]) ** 2 for i, t in enumerate(ti)]).mean()
    return e_se


def MSE(B, ti_inferred):
    """[summary]

    Args:
        M ([type]): [description]

    Returns:
        [type]: [description]
    """
    mse = np.array(
        [
            np.array([b * (ti - (t-1)) ** 2 for t, b in enumerate(B[i])]).sum()
            for i, ti in enumerate(ti_inferred)
        ]
    ).mean()

    return mse



''''''''''''''''''''''''''''''''''''


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run a simulation and don't ask.")
    parser.add_argument(
        "--save_dir", type=str, default="../Data/check/", dest="save_dir", help="save_dir"
    )
    parser.add_argument(
        "--graph", type=str, default="rrg", dest="graph", help="Type of random graph"
    )
    parser.add_argument(
        "--N", type=int, default=5000, dest="N", help="Number of individuals"
    )
    parser.add_argument(
        "--d", type=int, default=3, dest="d", help="degree of RRG"
    )
    parser.add_argument(
        "--n_sources", type=int, default=[-1], dest="n_sources", help="number of sources, pass as multiple arguments, e.g. 2 4 8", nargs="+",
    )
    parser.add_argument(
        "--delta", type=float, default=[-1], dest="delta", help="fraction of sources, pass as multiple arguments, e.g. 0.1 0.3 0.5", nargs="+",
    )
    parser.add_argument(
        "--lam", type=float, default=1., dest="lam", help="lambda"
    )
    parser.add_argument(
        "--nsim", type=int, default=25, dest="nsim", help="number of simulations"
    )
    parser.add_argument(
        "--print_it", type=int, default=0, dest="print_it", help="print every interation of BP"
    )
    parser.add_argument(
        "--n_obs", type=int, default=[-1], dest="n_obs", help="number of observations, pass as multiple arguments, e.g. 2 4 8",
        nargs="+",
    )
    parser.add_argument(
        "--rho", type=float, default=[-1], dest="rho", help="fraction of observations, pass as multiple arguments, e.g. 0.1 0.3 0.5",
        nargs="+",
    )
    parser.add_argument(
        "--seed", type=int, default=0, dest="seed", help="seed for the number generators"
    )
    parser.add_argument(
        "--mu", type=float, default=0., dest="mu", help="Recovery parameter, e.g. 0 for the SI model"
    )
    parser.add_argument(
        "--psus", type=float, default=0.5, dest="psus", help="Prior probability to be susceptible"
    )
    parser.add_argument(
        "--pseed0", type=int, default=0, dest="pseed0", help="If different from 0, pseed becomes really small (check)"
    )
    parser.add_argument(
        "--tol", type=float, default=1e-9, dest="tol", help="Tolerance of BP"
    )
    parser.add_argument(
        "--n_iter", type=int, default=2000, dest="n_iter", help="Number of max iterations of the algorithm"
    )
    parser.add_argument(
        "--iter_space", type=int, default=4, dest="iter_space", help="Space between saved iterations"
    )

    args = parser.parse_args()
    print("arguments:")
    print(args)

    if (args.n_sources==0):
        warnings.warn("YOU CANNOT HAVE ZERO SOURCES, THE SIMULATION WILL NEVER STOP!")
        sys.exit()
    save_dir = args.save_dir
    graph = args.graph
    path_save = Path(save_dir)
    if not path_save.exists():
        warnings.warn("SAVING FOLDER DOES NOT EXIST")
    

    N = args.N
    d = args.d
    if (args.n_sources[0] != -1 and args.delta[0] == -1): 
        flag_sources = 0
        sources_table = args.n_sources  
        pseed = np.array(sources_table) / N 
        if (args.pseed0) : pseed = pseed/100
    elif (args.n_sources[0] == -1 and args.delta[0] != -1): 
        flag_sources = 1
        sources_table = args.delta
        pseed = np.array(sources_table)
    else:  warnings.warn("YOU HAVE TO CHOOSE BETWEEN DELTA AND #SOURCES")
    lam = args.lam
    n_sim = args.nsim
    print_it = args.print_it
    if (args.n_obs[0] != -1 and args.rho[0] == -1): 
        flag_obs = 0
        obs_table = args.n_obs
    elif (args.n_obs[0] == -1 and args.rho[0] != -1): 
        flag_obs = 1
        obs_table = args.rho
    else:  warnings.warn("YOU HAVE TO CHOOSE BETWEEN RHO AND #OBS")

    if (graph == "rrg"):
        def generate_graph(n=N,d=d):
            return nx.random_regular_graph(n=n, d=d)
    elif (graph == "tree"):
        def generate_graph(n=N, r=d):
            return nx.full_rary_tree(r=r, n=n)
    else:  warnings.warn("GRAPH TYPE NOT ALLOWED")

    seed = args.seed  # setting seed everywhere for reproducibility TBD
    random.seed(seed) 
    np.random.seed(seed)

    mu = args.mu
    
    psus = args.psus  # prior to be S
    tol = args.tol
    n_iter = args.n_iter
    iter_space = args.iter_space

    data_obs = create_data_obs(flag_sources, flag_obs, n_sim, N, d, lam,n_iter, pseed)
    if (print_it): data_obs_it = create_data_obs_it(flag_sources, flag_obs, n_sim, N, d, lam,n_iter, pseed)
    else : data_obs_it = []
    t1 = time.time()
    t2 = time.time()
    for i_S, S in enumerate(sources_table):
        for i_M, M in enumerate(obs_table):
            for sim in range(n_sim):
                g = generate_graph(N, d)
                if (args.delta[0] == -1): status_nodes = generate_one_conf_ns_allI(g, lamb=lam, n_sources=S)
                else: status_nodes = generate_one_conf_delta_allI(g, lamb=lam, delta=S)
                T = len(status_nodes) - 1
                contacts = generate_contacts(g, T, lam)
                if (args.rho[0] == -1):
                    list_obs = generate_M_obs(status_nodes, M=M)              
                    list_obs_all = generate_M_obs(status_nodes, M=N)
                else:
                    list_obs = generate_obs(status_nodes, frac_obs=M)              
                    list_obs_all = generate_obs(status_nodes, frac_obs=1)

                f_rnd = FactorGraph(
                    N=N,
                    T=T,
                    contacts=contacts,
                    obs=[],
                    delta=pseed
                )
                f_informed = FactorGraph(
                    N=N,
                    T=T,
                    contacts=contacts,
                    obs=list_obs_all,
                    delta=pseed
                )

                sim_and_fill(f_rnd, data_obs, data_obs_it, list_obs, n_iter, tol, print_it, status_nodes, T, "rnd", M, S, iter_space, sim, flag_sources, flag_obs)
                sim_and_fill(f_informed, data_obs, data_obs_it, list_obs, n_iter, tol, print_it, status_nodes, T, "inf", M, S, iter_space, sim, flag_sources, flag_obs)
                print(
                    f"\r S: {i_S+1}/{len(sources_table)} - M: {i_M+1}/{len(obs_table)} - sim: {sim+1}/{n_sim} - time = {time.time()-t2:.2f} s - total time = {time.time()-t1:.0f} s"
                )
                t2 = time.time()
    
    dov = np.array(data_obs["ov0"]) - np.array(data_obs["mov0"])
    ovt = (np.array(data_obs["ov0"])  - np.array(data_obs["ov0_rnd"]) )/(1- np.array(data_obs["ov0_rnd"]))
    movt = (np.array(data_obs["mov0"]) - np.array(data_obs["mov0_rnd"]) )/(1- np.array(data_obs["mov0_rnd"]))
    dovt = ovt-movt
    dse = np.array(data_obs["se"]) - np.array(data_obs["mse"])
    if (print_it):
        dov_it = np.array(data_obs_it["ov0"]) - np.array(data_obs_it["mov0"])
        ovt_it = (np.array(data_obs_it["ov0"])  - np.array(data_obs_it["ov0_rnd"]) )/(1- np.array(data_obs_it["ov0_rnd"]))
        movt_it = (np.array(data_obs_it["mov0"]) - np.array(data_obs_it["mov0_rnd"]) )/(1- np.array(data_obs_it["mov0_rnd"]))
        dovt_it = ovt_it-movt_it
    if (flag_obs):
        if (flag_sources):
            data_list =  [ [data_obs["N"],data_obs["d"],data_obs["lam"],data_obs["n_sim"],data_obs["n_iter"],data_obs["pseed"],data_obs["delta"][i],data_obs["rho"][i],data_obs["init"][i],data_obs["ov0"][i],data_obs["mov0"][i],dov[i],data_obs["ov0_rnd"][i], data_obs["mov0_rnd"][i], ovt[i],movt[i],dovt[i],data_obs["e"][i],data_obs["se"][i],data_obs["mse"][i],dse[i],data_obs["logL"][i],data_obs["nI"][i], data_obs["T_table"][i], data_obs["sim"][i]]  for i, o in enumerate(data_obs["ov0"]) ] 
            if (print_it): data_list_it = [ [data_obs_it["N"],data_obs_it["d"],data_obs_it["lam"],data_obs_it["n_sim"],data_obs_it["n_iter"],data_obs_it["pseed"],data_obs_it["iter"][i], data_obs_it["delta"][i], data_obs_it["rho"][i],data_obs_it["init"][i],data_obs_it["ov0"][i],data_obs_it["mov0"][i],dov_it[i],data_obs_it["ov0_rnd"][i], data_obs_it["mov0_rnd"][i], ovt_it[i],movt_it[i],dovt_it[i],data_obs_it["e"][i],data_obs_it["logL"][i],data_obs_it["sim"][i],data_obs_it["converged"][i]]  for i, o in enumerate(data_obs_it["ov0"]) ]
            o = r"$\rho$"
            s = r"$\delta$"
        else:
            data_list =  [ [data_obs["N"],data_obs["d"],data_obs["lam"],data_obs["n_sim"],data_obs["n_iter"],data_obs["pseed"],data_obs["# sources"][i],data_obs["rho"][i],data_obs["init"][i],data_obs["ov0"][i],data_obs["mov0"][i],dov[i],data_obs["ov0_rnd"][i], data_obs["mov0_rnd"][i], ovt[i],movt[i],dovt[i],data_obs["e"][i],data_obs["se"][i],data_obs["mse"][i],dse[i],data_obs["logL"][i],data_obs["nI"][i], data_obs["T_table"][i], data_obs["sim"][i]]  for i, o in enumerate(data_obs["ov0"]) ] 
            if (print_it): data_list_it = [ [data_obs_it["N"],data_obs_it["d"],data_obs_it["lam"],data_obs_it["n_sim"],data_obs_it["n_iter"],data_obs_it["pseed"],data_obs_it["iter"][i], data_obs_it["# sources"][i], data_obs_it["rho"][i],data_obs_it["init"][i],data_obs_it["ov0"][i],data_obs_it["mov0"][i],dov_it[i],data_obs_it["ov0_rnd"][i], data_obs_it["mov0_rnd"][i], ovt_it[i],movt_it[i],dovt_it[i],data_obs_it["e"][i],data_obs_it["logL"][i],data_obs_it["sim"][i],data_obs_it["converged"][i]]  for i, o in enumerate(data_obs_it["ov0"]) ]
            o = r"$\rho$"
            s = "# sources"
    else:
        if (flag_sources):
            data_list =  [ [data_obs["N"],data_obs["d"],data_obs["lam"],data_obs["n_sim"],data_obs["n_iter"],data_obs["pseed"],data_obs["delta"][i],data_obs["# obs"][i],data_obs["init"][i],data_obs["ov0"][i],data_obs["mov0"][i],dov[i],data_obs["ov0_rnd"][i], data_obs["mov0_rnd"][i], ovt[i],movt[i],dovt[i],data_obs["e"][i],data_obs["se"][i],data_obs["mse"][i],dse[i],data_obs["logL"][i],data_obs["nI"][i], data_obs["T_table"][i], data_obs["sim"][i]]  for i, o in enumerate(data_obs["ov0"]) ] 
            if (print_it): data_list_it = [ [data_obs_it["N"],data_obs_it["d"],data_obs_it["lam"],data_obs_it["n_sim"],data_obs_it["n_iter"],data_obs_it["pseed"],data_obs_it["iter"][i], data_obs_it["delta"][i], data_obs_it["# obs"][i],data_obs_it["init"][i],data_obs_it["ov0"][i],data_obs_it["mov0"][i],dov_it[i],data_obs_it["ov0_rnd"][i], data_obs_it["mov0_rnd"][i], ovt_it[i],movt_it[i],dovt_it[i],data_obs_it["e"][i],data_obs_it["logL"][i],data_obs_it["sim"][i],data_obs_it["converged"][i]]  for i, o in enumerate(data_obs_it["ov0"]) ]
            o = "# obs"
            s = r"$\delta$"
        else:
            data_list =  [ [data_obs["N"],data_obs["d"],data_obs["lam"],data_obs["n_sim"],data_obs["n_iter"],data_obs["pseed"],data_obs["# sources"][i],data_obs["# obs"][i],data_obs["init"][i],data_obs["ov0"][i],data_obs["mov0"][i],dov[i],data_obs["ov0_rnd"][i], data_obs["mov0_rnd"][i], ovt[i],movt[i],dovt[i],data_obs["e"][i],data_obs["se"][i],data_obs["mse"][i],dse[i],data_obs["logL"][i],data_obs["nI"][i], data_obs["T_table"][i], data_obs["sim"][i]]  for i, o in enumerate(data_obs["ov0"]) ] 
            if (print_it): data_list_it = [ [data_obs_it["N"],data_obs_it["d"],data_obs_it["lam"],data_obs_it["n_sim"],data_obs_it["n_iter"],data_obs_it["pseed"],data_obs_it["iter"][i], data_obs_it["# sources"][i], data_obs_it["# obs"][i],data_obs_it["init"][i],data_obs_it["ov0"][i],data_obs_it["mov0"][i],dov_it[i],data_obs_it["ov0_rnd"][i], data_obs_it["mov0_rnd"][i], ovt_it[i],movt_it[i],dovt_it[i],data_obs_it["e"][i],data_obs_it["logL"][i],data_obs_it["sim"][i],data_obs_it["converged"][i]]  for i, o in enumerate(data_obs_it["ov0"]) ]
            o = "# obs"
            s = "# sources"

    s_N= r"$N$"
    s_d= r"$d$"
    s_l= r"$\lambda$"
    ov = r"$O_{t=0}$"
    mov = r"$MO_{t=0}$"
    ov_rnd = r"$O_{t=0,RND}$"
    mov_rnd = r"$MO_{t=0,RND}$"
    dov = r"$\delta O$"
    ovt = r"$\widetilde{O}_{t=0}$"
    movt = r"$\widetilde{MO}_{t=0}$"
    dovt = r"$\widetilde{\delta O}_{t=0}$"
    dse = r"$\delta SE$"
    se = "SE"
    mse = "MSE"

    data_frame = pd.DataFrame(data_list, columns=[s_N,s_d,s_l,"n_sim","n_iter","pseed",s, o,"init",ov,mov,dov,ov_rnd,mov_rnd,ovt,movt,dovt,"error",se,mse,dse,"logLikelihood","# iter", "T", "sim"])
    data_frame[s_N] = data_frame[s_N].astype(int)
    data_frame[s_d] = data_frame[s_d].astype(int)
    data_frame[s_l] = data_frame[s_l].astype(float)
    data_frame["n_sim"] = data_frame["n_sim"].astype(int)
    data_frame["n_iter"] = data_frame["n_iter"].astype(int)
    data_frame["pseed"] = data_frame["pseed"].astype(float)
    data_frame["init"] = data_frame["init"].astype(str)
    data_frame[ov] = data_frame[ov].astype(float)
    data_frame[mov] = data_frame[mov].astype(float)
    data_frame[dov] = data_frame[dov].astype(float)
    data_frame[ov_rnd] = data_frame[ov_rnd].astype(float)
    data_frame[mov_rnd] = data_frame[mov_rnd].astype(float)
    data_frame[ovt] = data_frame[ovt].astype(float)
    data_frame[movt] = data_frame[movt].astype(float)
    data_frame[dovt] = data_frame[dovt].astype(float)
    data_frame["error"] = data_frame["error"].astype(float)
    data_frame[se] = data_frame["SE"].astype(float)
    data_frame[mse] = data_frame["MSE"].astype(float)
    data_frame[dse] = data_frame[dse].astype(float)
    data_frame["logLikelihood"] = data_frame["logLikelihood"].astype(float)
    data_frame["# iter"] = data_frame["# iter"].astype(float)
    data_frame["T"] = data_frame["T"].astype(int)
    data_frame["sim"] = data_frame["sim"].astype(int)

    if (print_it):
        data_frame_it = pd.DataFrame(data_list_it, columns=[s_N,s_d,s_l,"n_sim","n_iter","pseed","iter", s, o,"init",ov,mov,dov,ov_rnd,mov_rnd,ovt,movt,dovt,"error","logLikelihood","sim","converged"])
        data_frame_it[s_N] = data_frame_it[s_N].astype(int)
        data_frame_it[s_d] = data_frame_it[s_d].astype(int)
        data_frame_it[s_l] = data_frame_it[s_l].astype(float)
        data_frame_it["n_sim"] = data_frame_it["n_sim"].astype(int)
        data_frame_it["n_iter"] = data_frame_it["n_iter"].astype(int)
        data_frame_it["pseed"] = data_frame_it["pseed"].astype(float)
        data_frame_it["iter"] = data_frame_it["iter"].astype(int)
        data_frame_it["init"] = data_frame_it["init"].astype(str)
        data_frame_it[ov] = data_frame_it[ov].astype(float)
        data_frame_it[mov] = data_frame_it[mov].astype(float)
        data_frame_it[dov] = data_frame_it[dov].astype(float)
        data_frame_it[ov_rnd] = data_frame_it[ov_rnd].astype(float)
        data_frame_it[mov_rnd] = data_frame_it[mov_rnd].astype(float)
        data_frame_it[ovt] = data_frame_it[ovt].astype(float)
        data_frame_it[movt] = data_frame_it[movt].astype(float)
        data_frame_it[dovt] = data_frame_it[dovt].astype(float)
        data_frame_it["error"] = data_frame_it["error"].astype(float)
        data_frame_it["logLikelihood"] = data_frame_it["logLikelihood"].astype(float)
        data_frame_it["sim"] = data_frame_it["sim"].astype(int)
        data_frame_it["converged"] = data_frame_it["converged"].astype(str)

    if (flag_obs):
        if (flag_sources):
            data_frame[o] = data_frame[o].astype(float)
            if (print_it): data_frame_it[o] = data_frame_it[o].astype(float)
            data_frame[s] = data_frame[s].astype(float)
            if (print_it): data_frame_it[s] = data_frame_it[s].astype(float)
            file_name = "data_{}_N{}_d{}_deltaMax{:.4f}_lam{:.2f}_rhoMax{:.3f}.xz".format(graph, N, d, S,lam,M)
        else:
            data_frame[o] = data_frame[o].astype(float)
            if (print_it): data_frame_it[o] = data_frame_it[o].astype(float)
            data_frame[s] = data_frame[s].astype(int)
            if (print_it): data_frame_it[s] = data_frame_it[s].astype(int)
            file_name = "data_{}_N{}_d{}_nsMax{}_lam{:.2f}_rhoMax{:.3f}.xz".format(graph, N, d, S,lam,M)
    else:
        if (flag_sources):
            data_frame[o] = data_frame[o].astype(int)
            if (print_it): data_frame_it[o] = data_frame_it[o].astype(int)
            data_frame[s] = data_frame[s].astype(float)
            if (print_it): data_frame_it[s] = data_frame_it[s].astype(float)
            file_name = "data_{}_N{}_d{}_deltaMax{:.4f}_lam{:.2f}_nobsMax{}.xz".format(graph, N, d, S,lam,M)

        else:
            data_frame[o] = data_frame[o].astype(int)
            if (print_it): data_frame_it[o] = data_frame_it[o].astype(int)
            data_frame[s] = data_frame[s].astype(int)
            if (print_it): data_frame_it[s] = data_frame_it[s].astype(int)
            file_name = "data_{}_N{}_d{}_nsMax{}_lam{:.2f}_nobsMax{}.xz".format(graph, N, d, S,lam,M)

    if (print_it) : saveObj = (data_frame, data_frame_it, np.array(data_obs["marginal0"]))
    else : saveObj = (data_frame,np.array(data_obs["marginal0"]))

    with lzma.open(save_dir + file_name, "wb") as f:
        pickle.dump(saveObj, f)