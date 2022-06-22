import pandas as pd

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
"""""" """""" """""" """""" """""" """"""


#def generate_one_conf(g, lamb=0.05, T=10, percentage_infected=0.01):
#    N = g.number_of_nodes()
#    model = ep.SIModel(g)
#    cfg = mc.Configuration()
#    cfg.add_model_parameter("beta", lamb)  # infection rate
#    # cfg.add_model_parameter('gamma', 0.0) # recovery rate
#    cfg.add_model_parameter("percentage_infected", percentage_infected)
#    model.set_initial_status(cfg)
#    iterations = model.iteration_bunch(T, node_status=True)
#    status_nodes = np.zeros((T, N))
#    for t in range(0, T):
#        for node_i in iterations[t]["status"].keys():
#            status_nodes[t:T, node_i] = iterations[t]["status"][node_i]
#    return np.array(status_nodes)


def generate_contacts(G, t_limit, lambda_, p_edge=1):
    contacts = []
    for t in range(t_limit):
        for e in G.edges():
            if random.random() <= p_edge:
                contacts.append((e[0], e[1], t, lambda_))
                contacts.append((e[1], e[0], t, lambda_))
    # contacts = np.array(contacts, dtype=[("t","f8"), ("i","f8"), ("j","f8"), ("lam", "f8")])
    # contacts.sort(axis=0, order=("tù","i","j"))
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
    fI = 1
    return obs_sim, fI


def generate_snapshot_obs(conf, frac_obs=0.0):
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


def iter_print(t, e, f):
    print(f"", end="\r", flush=True)


def create_data_obs(flag_sources, flag_obs, n_sim, N, d, lam, n_iter, pseed, Delta):
    data_obs = {}
    data_obs["N"] = N
    data_obs["d"] = d
    data_obs["lam"] = lam
    data_obs["Delta"] = Delta
    data_obs["n_sim"] = n_sim
    data_obs["n_iter"] = n_iter
    data_obs["pseed"] = pseed
    if flag_obs:
        data_obs["rho"] = []
    else:
        data_obs["# obs"] = []
    if flag_sources:
        data_obs["delta"] = []
    else:
        data_obs["# sources"] = []
    data_obs["init"] = []
    data_obs["ov0"] = []  # overlap with the ground truth at t=0, n_T x n_M x n_sim
    data_obs["ov0_rnd"] = []  # overlap with the ground truth at t=0, n_T x n_M x n_sim
    data_obs["mov0"] = []  # mean overlap at t=0, n_T x n_M x n_sim
    data_obs["mov0_rnd"] = []  # overlap with the ground truth at t=0, n_T x n_M x n_sim
    data_obs["ovT"] = []  # overlap with the ground truth at t=0, n_T x n_M x n_sim
    data_obs["ovT_rnd"] = []  # overlap with the ground truth at t=0, n_T x n_M x n_sim
    data_obs["movT"] = []  # mean overlap at t=0, n_T x n_M x n_sim
    data_obs["movT_rnd"] = []  # overlap with the ground truth at t=0, n_T x n_M x n_sim
    data_obs["se"] = []
    data_obs["mse"] = []
    data_obs["se_rnd"] = []
    data_obs["mse_rnd"] = []
    data_obs["nI"] = []
    data_obs["e"] = []
    data_obs["logL"] = []
    data_obs["marginal0"] = []
    data_obs["sim"] = []
    data_obs["T_table"] = []
    #data_obs["T0_table"] = []
    data_obs["fI"] = []
    # data_obs["obs_table"] = []#np.zeros((n_M, n_sim, N))

    return data_obs


def fill_data_obs(
    data_obs, f, status_nodes, TO, it, e, init, M, S, sim, flag_sources, flag_obs, fI, T, Delta
):
    Bs = f.marginals()
    MI0 = Bs[:, 0]
    MS0 = 1-MI0
    MR0 = np.zeros_like(MS0)
    M0 = np.array([ MS0, MI0, MR0])
    #Compute general marginals:
    pS = 1 - np.cumsum(Bs,axis=1)
    pI = np.array([ np.array([ sum(b[1+t-Delta:1+t]) if t>=Delta else sum(b[:t+1]) for t in range(T+2)]) for b in Bs])
    pR = 1 - pS - pI
    #Take last time
    ###Ms0 = np.cumsum(Bs,axis=1)[:,T]
    MST = pS[:,T]
    MIT = pI[:,T]
    MRT = pR[:,T]
    MT = np.array([ MST, MIT, MRT])
    Ss = status_nodes
    x0_inf = np.argmax(M0,axis=0)
    xT_inf = np.argmax(MT,axis=0)
    ti_str = ti_star(Ss)

    data_obs["T_table"].append(T)
    #data_obs["T0_table"].append(TO)
    # data_obs["obs_table"].append( obs_table)
    if flag_obs:
        data_obs["rho"].append(M)
    else:
        data_obs["# obs"].append(M)
    if flag_sources:
        data_obs["delta"].append(S)
    else:
        data_obs["# sources"].append(S)
    data_obs["sim"].append(sim)

    data_obs["init"].append(init)
    data_obs["fI"].append(fI)
    data_obs["ov0"].append(E_overlap(Ss[0], x0_inf))
    data_obs["ov0_rnd"].append(E_overlap_rnd(Ss[0], M0))
    data_obs["mov0"].append(M_overlap(M0))
    data_obs["mov0_rnd"].append(M_overlap_rnd(M0))
    data_obs["ovT"].append(E_overlap(Ss[T], xT_inf))
    data_obs["ovT_rnd"].append(E_overlap_rnd(Ss[T], MT))
    data_obs["movT"].append(M_overlap(MT))
    data_obs["movT_rnd"].append(M_overlap_rnd(MT))
    ti_inf = ti_inferred(Bs)
    ti_rnd = ti_random(Bs)
    data_obs["se"].append(E_SE(ti_str, ti_inf))
    data_obs["mse"].append(MSE(Bs, ti_inf))
    data_obs["se_rnd"].append(E_SE(ti_str, ti_rnd))
    data_obs["mse_rnd"].append(MSE(Bs, ti_rnd))
    data_obs["nI"].append(it)
    data_obs["e"].append(e)
    data_obs["logL"].append(f.loglikelihood())
    data_obs["marginal0"].append(MT)


def create_data_obs_it(flag_sources, flag_obs, n_sim, N, d, lam, n_iter, pseed):
    data_obs_it = {}

    data_obs_it["N"] = N
    data_obs_it["d"] = d
    data_obs_it["lam"] = lam
    data_obs_it["n_sim"] = n_sim
    data_obs_it["n_iter"] = n_iter
    data_obs_it["pseed"] = pseed
    data_obs_it["init"] = []
    data_obs_it["fI"] = []
    data_obs_it["sim"] = []
    data_obs_it["converged"] = []
    if flag_obs:
        data_obs_it["rho"] = []
    else:
        data_obs_it["# obs"] = []
    if flag_sources:
        data_obs_it["delta"] = []
    else:
        data_obs_it["# sources"] = []
    data_obs_it["iter"] = []
    data_obs_it["ov0"] = []  # overlap with the ground truth at t=0, n_T x n_M x n_sim
    data_obs_it[
        "ov0_rnd"
    ] = []  # overlap with the ground truth at t=0, n_T x n_M x n_sim
    data_obs_it["mov0"] = []  # mean overlap at t=0, n_T x n_M x n_sim
    data_obs_it[
        "mov0_rnd"
    ] = []  # overlap with the ground truth at t=0, n_T x n_M x n_sim
    data_obs_it["e"] = []
    data_obs_it["logL"] = []

    return data_obs_it


def fill_data_obs_it(
    data_obs_it,
    f,
    status_nodes,
    TO,
    it,
    e,
    init,
    M,
    S,
    tol,
    n_iter,
    n_it_print,
    sim,
    flag_sources,
    flag_obs,
    fI,
    T
):
    Bs = f.marginals()
    #Ms0 = Bs[:, 0]
    Ms0 = np.cumsum(Bs,axis=1)[:,T]
    Ss = status_nodes

    data_obs_it["init"].append(init)
    data_obs_it["fI"].append(fI)
    data_obs_it["sim"].append(sim)
    if flag_obs:
        data_obs_it["rho"].append(M)
    else:
        data_obs_it["# obs"].append(M)
    if flag_sources:
        data_obs_it["delta"].append(S)
    else:
        data_obs_it["# sources"].append(S)
    data_obs_it["iter"].append(it)
    data_obs_it["ov0"].append(E_overlap(Ss[0], (Ms0 > 0.5)))
    data_obs_it["ov0_rnd"].append(E_overlap_rnd(Ss[0], Ms0))
    data_obs_it["mov0"].append(M_overlap(Ms0))
    data_obs_it["mov0_rnd"].append(M_overlap_rnd(Ms0))
    data_obs_it["e"].append(e)
    if e < tol:
        data_obs_it["converged"].extend(["yes"] * n_it_print)
    elif it == n_iter:
        data_obs_it["converged"].extend(["no"] * n_it_print)
    data_obs_it["logL"].append(f.loglikelihood())


def sim_and_fill(
    f,
    data_obs,
    data_obs_it,
    list_obs,
    n_iter,
    tol,
    print_it,
    status_nodes,
    TO,
    init,
    M,
    S,
    iter_space,
    sim,
    flag_sources,
    flag_obs,
    fI,
    T,
    Delta
):
    tol2 = 1e-2
    it_max = 10000
    for it in range(n_iter):
        e0 = f.iterate()
        if e0 < tol:
            break
    if e0 > tol:
        warnings.warn("Warning... Initialization is not converging")
    f.reset_obs(list_obs)

    e = np.nan
    n_it_print = 1
    if print_it:
        fill_data_obs_it(
            data_obs_it,
            f,
            status_nodes,
            TO,
            0,
            e,
            init,
            M,
            S,
            tol,
            n_iter,
            n_it_print,
            sim,
            flag_sources,
            flag_obs,
            fI,
            T
        )
    for it in range(n_iter):
        e = f.iterate()
        if print_it and (
            ((it + 1) % iter_space == 0) or (e < tol or (it + 1) == n_iter)
        ):
            n_it_print = n_it_print + 1
            fill_data_obs_it(
                data_obs_it,
                f,
                status_nodes,
                TO,
                it + 1,
                e,
                init,
                M,
                S,
                tol,
                n_iter,
                n_it_print,
                sim,
                flag_sources,
                flag_obs,
                fI,
                T
            )
        if e < tol:
            break
    while (e > tol) and (e < tol2):
        it = it + 1
        e = f.iterate()
        if print_it and ((it + 1) % iter_space == 0):
            n_it_print = n_it_print + 1
            fill_data_obs_it(
                data_obs_it,
                f,
                status_nodes,
                TO,
                it + 1,
                e,
                init,
                M,
                S,
                tol,
                n_iter,
                n_it_print,
                sim,
                flag_sources,
                flag_obs,
                fI,
                T
            )
        if it == it_max:
            break
    fill_data_obs(
        data_obs, f, status_nodes, TO, it + 1, e, init, M, S, sim, flag_sources, flag_obs, fI, T, Delta
    )


#def generate_one_conf_ns_allI(g, lamb, n_sources):
#    """[summary]
#
#    Args:
#        g ([type]): [description]
#        lamb (float, optional): [description]. Defaults to 0.05.
#        T (int, optional): [description]. Defaults to 10.
#        percentage_infected (float, optional): [description]. Defaults to 0.01.
#
#    Returns:
#        [type]: [description]
#    """
#    N = g.number_of_nodes()
#    model = ep.SIModel(g)
#    cfg = mc.Configuration()
#    cfg.add_model_parameter("beta", lamb)  # infection rate
#    # cfg.add_model_parameter('gamma', 0.0) # recovery rate
#    infected_nodes = random.sample(range(N), n_sources)
#    cfg.add_model_initial_configuration("Infected", infected_nodes)
#    model.set_initial_status(cfg)
#    status_nodes = []
#    status_nodes_t = np.zeros(N, dtype=int)
#    t = 0
#    while sum(status_nodes_t) < N:
#        iteration = model.iteration(node_status=True)
#        for node_i in iteration["status"].keys():
#            status_nodes_t[node_i] = iteration["status"][node_i]
#        status_nodes.append(status_nodes_t.copy())
#        t = t + 1
#
#    return np.array(status_nodes)
#
#
#def generate_one_conf_delta_allI(g, lamb, delta):
#    """[summary]
#
#    Args:
#        g ([type]): [description]
#        lamb (float, optional): [description]. Defaults to 0.05.
#        T (int, optional): [description]. Defaults to 10.
#        percentage_infected (float, optional): [description]. Defaults to 0.01.
#
#    Returns:
#        [type]: [description]
#    """
#    N = g.number_of_nodes()
#    model = ep.SIModel(g)
#    cfg = mc.Configuration()
#    cfg.add_model_parameter("beta", lamb)  # infection rate
#    # cfg.add_model_parameter('gamma', 0.0) # recovery rate
#    infected_nodes = []
#    while len(infected_nodes) == 0:
#        for i in range(N):
#            if np.random.random() < delta:
#                infected_nodes.append(i)
#    cfg.add_model_initial_configuration("Infected", infected_nodes)
#    model.set_initial_status(cfg)
#    status_nodes = []
#    status_nodes_t = np.zeros(N, dtype=int)
#    t = 0
#    while sum(status_nodes_t) < N:
#        iteration = model.iteration(node_status=True)
#        for node_i in iteration["status"].keys():
#            status_nodes_t[node_i] = iteration["status"][node_i]
#        status_nodes.append(status_nodes_t.copy())
#        t = t + 1
#
#    return np.array(status_nodes)

def simulate_one_detSIR(G, p_inf = 0.01, mask = ["SI"]):
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
    N = len(S[0])
    T = len(S)-1
    ti = np.zeros(N)
    for i in range(N):
        t_inf = np.nonzero(S[:, i] == 1)[0]
        if len(t_inf) == 0:
            ti[i] = T
        else:
            ti[i] = t_inf[0] - 1
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
        [np.array([(t - 1) * bt for t, bt in enumerate(b)]).sum() for b in B]
    )
    # return np.array([ np.array([ t*bt for t,bt in enumerate(b) if t != T ]).sum() for i,b in enumerate(B)])

def ti_random(B):
    """[summary]

    Args:
        M ([type]): [description]

    Returns:
        [type]: [description]
    """
    T = len(B[0])-2
    b_mean = B.mean(axis=0)
    ti =  np.array(
        [ t * b_mean[i]  for i,t  in enumerate(range(-1,T+1))]
    ).sum()
    return np.full(B.shape[0],ti)


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


def M_overlap(MT, axis=(-1)):
    """[summary]

    Args:
        M ([type]): [description]

    Returns:
        [type]: [description]
    """
    M0 = np.maximum(MT[0],MT[1])
    return np.maximum(M0, MT[2]).mean(axis=axis)


def E_overlap_rnd(conf, MT, axis=-1):
    """
    [summary]

    Args:
        conf ([type]): [description]
    Returns:
        [type]: [description]
    """
    x = np.argmax(np.mean(MT,axis=1))
    return (conf == x).mean()


def M_overlap_rnd(MT, axis=-1):
    """
    [summary]

    Args:
        conf ([type]): [description]
    Returns:
        [type]: [description]
    """
    m1 = np.maximum(MT[0].mean(axis=axis), MT[1].mean(axis=axis))
    return np.maximum(m1, MT[2].mean(axis=axis))


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
            np.array([b * (ti - (t - 1)) ** 2 for t, b in enumerate(B[i])]).sum()
            for i, ti in enumerate(ti_inferred)
        ]
    ).mean()

    return mse


"""""" """""" """""" """""" """""" """"""


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run a simulation and don't ask.")
    parser.add_argument(
        "--save_dir",
        type=str,
        default="../Data/check/",
        dest="save_dir",
        help="save_dir",
    )
    parser.add_argument(
        "--graph", type=str, default="rrg", dest="graph", help="Type of random graph"
    )
    parser.add_argument(
        "--N", type=int, default=5000, dest="N", help="Number of individuals"
    )
    parser.add_argument("--d", type=int, default=3, dest="d", help="degree of RRG")
    parser.add_argument(
        "--n_sources",
        type=int,
        default=[-1],
        dest="n_sources",
        help="number of sources, pass as multiple arguments, e.g. 2 4 8",
        nargs="+",
    )
    parser.add_argument(
        "--delta",
        type=float,
        default=[-1],
        dest="delta",
        help="fraction of sources, pass as multiple arguments, e.g. 0.1 0.3 0.5",
        nargs="+",
    )
    parser.add_argument("--lam", type=float, default=1.0, dest="lam", help="lambda")
    parser.add_argument(
        "--nsim", type=int, default=25, dest="nsim", help="number of simulations"
    )
    parser.add_argument(
        "--print_it",
        type=int,
        default=0,
        dest="print_it",
        help="print every interation of BP",
    )
    parser.add_argument(
        "--n_obs",
        type=int,
        default=[-1],
        dest="n_obs",
        help="number of observations, pass as multiple arguments, e.g. 2 4 8",
        nargs="+",
    )
    parser.add_argument(
        "--rho",
        type=float,
        default=[-1],
        dest="rho",
        help="fraction of observations, pass as multiple arguments, e.g. 0.1 0.3 0.5",
        nargs="+",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=0,
        dest="seed",
        help="seed for the number generators",
    )
    parser.add_argument(
        "--mu",
        type=float,
        default=0.0,
        dest="mu",
        help="Recovery parameter, e.g. 0 for the SI model",
    )
    parser.add_argument(
        "--psus",
        type=float,
        default=0.5,
        dest="psus",
        help="Prior probability to be susceptible",
    )
    parser.add_argument(
        "--pseed0",
        type=int,
        default=0,
        dest="pseed0",
        help="If different from 0, pseed becomes really small (check)",
    )
    parser.add_argument(
        "--tol", type=float, default=1e-9, dest="tol", help="Tolerance of BP"
    )
    parser.add_argument(
        "--n_iter",
        type=int,
        default=2000,
        dest="n_iter",
        help="Number of max iterations of the algorithm",
    )
    parser.add_argument(
        "--iter_space",
        type=int,
        default=4,
        dest="iter_space",
        help="Space between saved iterations",
    )

    parser.add_argument(
        "--snap",
        type=int,
        default=0,
        dest="snap",
        help="Snapshot observation",
    )
    parser.add_argument(
        "--Delta",
        type=int,
        default=10,
        dest="Delta",
        help="Lenght of mask array",
    )

    args = parser.parse_args()
    print("arguments:")
    print(args)

    if args.n_sources == 0:
        warnings.warn("YOU CANNOT HAVE ZERO SOURCES, THE SIMULATION WILL NEVER STOP!")
        sys.exit()
    save_dir = args.save_dir
    graph = args.graph
    path_save = Path(save_dir)
    if not path_save.exists():
        warnings.warn("SAVING FOLDER DOES NOT EXIST")

    N = args.N
    d = args.d
    if args.n_sources[0] != -1 and args.delta[0] == -1:
        flag_sources = 0
        sources_table = args.n_sources
        pseed = np.array(sources_table) / N
        if args.pseed0:
            pseed = pseed / 100
    elif args.n_sources[0] == -1 and args.delta[0] != -1:
        flag_sources = 1
        sources_table = args.delta
        pseed = np.array(sources_table)
    else:
        warnings.warn("YOU HAVE TO CHOOSE BETWEEN DELTA AND #SOURCES")
    lam = args.lam
    n_sim = args.nsim
    print_it = args.print_it
    if args.n_obs[0] != -1 and args.rho[0] == -1:
        flag_obs = 0
        obs_table = args.n_obs
    elif args.n_obs[0] == -1 and args.rho[0] != -1:
        flag_obs = 1
        obs_table = args.rho
    else:
        warnings.warn("YOU HAVE TO CHOOSE BETWEEN RHO AND #OBS")

    if graph == "rrg":

        def generate_graph(n=N, d=d):
            return nx.random_regular_graph(n=n, d=d)

    elif graph == "tree":

        def generate_graph(n=N, r=d):
            return nx.full_rary_tree(r=r, n=n)

    else:
        warnings.warn("GRAPH TYPE NOT ALLOWED")

    seed = args.seed  # setting seed everywhere for reproducibility TBD
    random.seed(seed)
    np.random.seed(seed)

    mu = args.mu

    psus = args.psus  # prior to be S
    tol = args.tol
    n_iter = args.n_iter
    iter_space = args.iter_space
    snap = args.snap
    Delta = args.Delta
    mask = [1]*Delta

    data_obs = create_data_obs(flag_sources, flag_obs, n_sim, N, d, lam, n_iter, pseed, Delta)
    if print_it:
        data_obs_it = create_data_obs_it(
            flag_sources, flag_obs, n_sim, N, d, lam, n_iter, pseed
        )
    else:
        data_obs_it = []
    t1 = time.time()
    t2 = time.time()
    for i_S, S in enumerate(sources_table):
        for i_M, M in enumerate(obs_table):
            for sim in range(n_sim):
                #g = generate_graph(N, d)
                if args.delta[0] == -1:
                    #status_nodes = generate_one_conf_ns_allI(g, lamb=lam, n_sources=S)
                    pass
                else:
                    G = nx.random_regular_graph(d=d, n=N)
                    for (u,v) in G.edges():
                        G.edges[u,v]['lambda'] = lam
                    status_nodes = simulate_one_detSIR(G, p_inf = S, mask = mask)
                T = len(status_nodes) - 1
                contacts = generate_contacts(G, T, lam)
                if (snap):
                    list_obs, fI, TO = generate_snapshot_obs(status_nodes, frac_obs=M)
                    list_obs_all, _ = generate_obs(status_nodes, frac_obs=1)
                else:
                    if args.rho[0] == -1:
                        list_obs, _ = generate_M_obs(status_nodes, M=M)
                        list_obs_all, _ = generate_M_obs(status_nodes, M=N)
                    else:
                        list_obs, _ = generate_obs(status_nodes, frac_obs=M)
                        list_obs_all, _ = generate_obs(status_nodes, frac_obs=1)
                    TO=T
                    fI = np.mean(status_nodes[-1]==2)

                f_rnd = FactorGraph(N=N, T=T, contacts=contacts, obs=[], delta=S, mask=mask)
                f_informed = FactorGraph(
                    N=N, T=T, contacts=contacts, obs=list_obs_all, delta=S, mask=mask
                )

                sim_and_fill(
                    f_rnd,
                    data_obs,
                    data_obs_it,
                    list_obs,
                    n_iter,
                    tol,
                    print_it,
                    status_nodes,
                    TO,
                    "rnd",
                    M,
                    S,
                    iter_space,
                    sim,
                    flag_sources,
                    flag_obs,
                    fI,
                    T,
                    Delta
                )
                sim_and_fill(
                    f_informed,
                    data_obs,
                    data_obs_it,
                    list_obs,
                    n_iter,
                    tol,
                    print_it,
                    status_nodes,
                    TO,
                    "inf",
                    M,
                    S,
                    iter_space,
                    sim,
                    flag_sources,
                    flag_obs,
                    fI,
                    T,
                    Delta
                )
                print(
                    f"\r S: {i_S+1}/{len(sources_table)} - M: {i_M+1}/{len(obs_table)} - sim: {sim+1}/{n_sim} - time = {time.time()-t2:.2f} s - total time = {time.time()-t1:.0f} s"
                )
                t2 = time.time()

    dov0 = np.array(data_obs["ov0"]) - np.array(data_obs["mov0"])
    ov0t = (np.array(data_obs["ov0"]) - np.array(data_obs["ov0_rnd"])) / (
        1 - np.array(data_obs["ov0_rnd"])
    )
    mov0t = (np.array(data_obs["mov0"]) - np.array(data_obs["mov0_rnd"])) / (
        1 - np.array(data_obs["mov0_rnd"])
    )
    dov0t = ov0t - mov0t
    dovT = np.array(data_obs["ovT"]) - np.array(data_obs["movT"])
    ovTt = (np.array(data_obs["ovT"]) + 1e-9 - np.array(data_obs["ovT_rnd"])) / (
        1 + 1e-9 - np.array(data_obs["ovT_rnd"])
    )
    movTt = (np.array(data_obs["movT"]) + 1e-9 - np.array(data_obs["movT_rnd"])) / (
        1 + 1e-9 - np.array(data_obs["movT_rnd"])
    )
    dovTt = ovTt - movTt
    dse = np.array(data_obs["se"]) - np.array(data_obs["mse"])
    Rse = (np.array(data_obs["se_rnd"]) - np.array(data_obs["se"])) / np.array(data_obs["se_rnd"])
    Rmse = (np.array(data_obs["mse_rnd"]) - np.array(data_obs["mse"])) / np.array(data_obs["mse_rnd"])
    dRse = Rse-Rmse
    if print_it: #TO BE CHANGED!!!!
        dov_it = np.array(data_obs_it["ov0"]) - np.array(data_obs_it["mov0"])
        ovt_it = (np.array(data_obs_it["ov0"]) - np.array(data_obs_it["ov0_rnd"])) / (
            1 - np.array(data_obs_it["ov0_rnd"])
        )
        movt_it = (
            np.array(data_obs_it["mov0"]) - np.array(data_obs_it["mov0_rnd"])
        ) / (1 - np.array(data_obs_it["mov0_rnd"]))
        dovt_it = ovt_it - movt_it
    if flag_obs:
        if flag_sources:
            data_list = [
                [
                    data_obs["N"],
                    data_obs["d"],
                    data_obs["lam"],
                    data_obs["Delta"],
                    data_obs["n_sim"],
                    data_obs["n_iter"],
                    #data_obs["pseed"],
                    data_obs["delta"][i],
                    data_obs["rho"][i],
                    data_obs["init"][i],
                    data_obs["ov0"][i],
                    data_obs["mov0"][i],
                    dov0[i],
                    data_obs["ov0_rnd"][i],
                    data_obs["mov0_rnd"][i],
                    ov0t[i],
                    mov0t[i],
                    dov0t[i],
                    data_obs["ovT"][i],
                    data_obs["movT"][i],
                    dovT[i],
                    data_obs["ovT_rnd"][i],
                    data_obs["movT_rnd"][i],
                    ovTt[i],
                    movTt[i],
                    dovTt[i],
                    data_obs["e"][i],
                    data_obs["se"][i],
                    data_obs["mse"][i],
                    data_obs["se_rnd"][i],
                    data_obs["mse_rnd"][i],
                    dse[i],
                    Rse[i],
                    Rmse[i],
                    dRse[i],
                    data_obs["logL"][i],
                    data_obs["nI"][i],
                    data_obs["T_table"][i],
                    data_obs["sim"][i],
                    data_obs["fI"][i],
                ]
                for i, o in enumerate(data_obs["ov0"])
            ]
            if print_it: #TO BE CHANGED!!!
                data_list_it = [
                    [
                        data_obs_it["N"],
                        data_obs_it["d"],
                        data_obs_it["lam"],
                        data_obs_it["n_sim"],
                        data_obs_it["n_iter"],
                        #data_obs_it["pseed"],
                        data_obs_it["iter"][i],
                        data_obs_it["delta"][i],
                        data_obs_it["rho"][i],
                        data_obs_it["init"][i],
                        data_obs_it["ov0"][i],
                        data_obs_it["mov0"][i],
                        dov_it[i],
                        data_obs_it["ov0_rnd"][i],
                        data_obs_it["mov0_rnd"][i],
                        ovt_it[i],
                        movt_it[i],
                        dovt_it[i],
                        data_obs_it["e"][i],
                        data_obs_it["logL"][i],
                        data_obs_it["sim"][i],
                        data_obs_it["fI"][i],
                        data_obs_it["converged"][i],
                    ]
                    for i, o in enumerate(data_obs_it["ov0"])
                ]
            o = r"$\rho$"
            s = r"$\delta$"
        else:
            data_list = [
                [
                    data_obs["N"],
                    data_obs["d"],
                    data_obs["lam"],
                    data_obs["Delta"],
                    data_obs["n_sim"],
                    data_obs["n_iter"],
                    #data_obs["pseed"],
                    data_obs["# sources"][i],
                    data_obs["rho"][i],
                    data_obs["init"][i],
                    data_obs["ov0"][i],
                    data_obs["mov0"][i],
                    dov0[i],
                    data_obs["ov0_rnd"][i],
                    data_obs["mov0_rnd"][i],
                    ov0t[i],
                    mov0t[i],
                    dov0t[i],
                    data_obs["ovT"][i],
                    data_obs["movT"][i],
                    dovT[i],
                    data_obs["ovT_rnd"][i],
                    data_obs["movT_rnd"][i],
                    ovTt[i],
                    movTt[i],
                    dovTt[i],
                    data_obs["e"][i],
                    data_obs["se"][i],
                    data_obs["mse"][i],
                    data_obs["se_rnd"][i],
                    data_obs["mse_rnd"][i],
                    dse[i],
                    Rse[i],
                    Rmse[i],
                    dRse[i],
                    data_obs["logL"][i],
                    data_obs["nI"][i],
                    data_obs["T_table"][i],
                    data_obs["sim"][i],
                    data_obs["fI"][i],
                ]
                for i, o in enumerate(data_obs["ov0"])
            ]
            if print_it:
                data_list_it = [
                    [
                        data_obs_it["N"],
                        data_obs_it["d"],
                        data_obs_it["lam"],
                        data_obs_it["n_sim"],
                        data_obs_it["n_iter"],
                        #data_obs_it["pseed"],
                        data_obs_it["iter"][i],
                        data_obs_it["# sources"][i],
                        data_obs_it["rho"][i],
                        data_obs_it["init"][i],
                        data_obs_it["ov0"][i],
                        data_obs_it["mov0"][i],
                        dov_it[i],
                        data_obs_it["ov0_rnd"][i],
                        data_obs_it["mov0_rnd"][i],
                        ovt_it[i],
                        movt_it[i],
                        dovt_it[i],
                        data_obs_it["e"][i],
                        data_obs_it["logL"][i],
                        data_obs_it["sim"][i],
                        data_obs_it["fI"][i],
                        data_obs_it["converged"][i],
                    ]
                    for i, o in enumerate(data_obs_it["ov0"])
                ]
            o = r"$\rho$"
            s = "# sources"
    else:
        if flag_sources:
            data_list = [
                [
                    data_obs["N"],
                    data_obs["d"],
                    data_obs["lam"],
                    data_obs["Delta"],
                    data_obs["n_sim"],
                    data_obs["n_iter"],
                    #data_obs["pseed"],
                    data_obs["delta"][i],
                    data_obs["# obs"][i],
                    data_obs["init"][i],
                    data_obs["ov0"][i],
                    data_obs["mov0"][i],
                    dov0[i],
                    data_obs["ov0_rnd"][i],
                    data_obs["mov0_rnd"][i],
                    ov0t[i],
                    mov0t[i],
                    dov0t[i],
                    data_obs["ovT"][i],
                    data_obs["movT"][i],
                    dovT[i],
                    data_obs["ovT_rnd"][i],
                    data_obs["movT_rnd"][i],
                    ovTt[i],
                    movTt[i],
                    dovTt[i],
                    data_obs["e"][i],
                    data_obs["se"][i],
                    data_obs["mse"][i],
                    data_obs["se_rnd"][i],
                    data_obs["mse_rnd"][i],
                    dse[i],
                    Rse[i],
                    Rmse[i],
                    dRse[i],
                    data_obs["logL"][i],
                    data_obs["nI"][i],
                    data_obs["T_table"][i],
                    data_obs["sim"][i],
                    data_obs["fI"][i],
                ]
                for i, o in enumerate(data_obs["ov0"])
            ]
            if print_it:
                data_list_it = [
                    [
                        data_obs_it["N"],
                        data_obs_it["d"],
                        data_obs_it["lam"],
                        data_obs_it["n_sim"],
                        data_obs_it["n_iter"],
                        #data_obs_it["pseed"],
                        data_obs_it["iter"][i],
                        data_obs_it["delta"][i],
                        data_obs_it["# obs"][i],
                        data_obs_it["init"][i],
                        data_obs_it["ov0"][i],
                        data_obs_it["mov0"][i],
                        dov_it[i],
                        data_obs_it["ov0_rnd"][i],
                        data_obs_it["mov0_rnd"][i],
                        ovt_it[i],
                        movt_it[i],
                        dovt_it[i],
                        data_obs_it["e"][i],
                        data_obs_it["logL"][i],
                        data_obs_it["sim"][i],
                        data_obs_it["fI"][i],
                        data_obs_it["converged"][i],
                    ]
                    for i, o in enumerate(data_obs_it["ov0"])
                ]
            o = "# obs"
            s = r"$\delta$"
        else:
            data_list = [
                [
                    data_obs["N"],
                    data_obs["d"],
                    data_obs["lam"],
                    data_obs["Delta"],
                    data_obs["n_sim"],
                    data_obs["n_iter"],
                    #data_obs["pseed"],
                    data_obs["# sources"][i],
                    data_obs["# obs"][i],
                    data_obs["init"][i],
                    data_obs["ov0"][i],
                    data_obs["mov0"][i],
                    dov0[i],
                    data_obs["ov0_rnd"][i],
                    data_obs["mov0_rnd"][i],
                    ov0t[i],
                    mov0t[i],
                    dov0t[i],
                    data_obs["ovT"][i],
                    data_obs["movT"][i],
                    dovT[i],
                    data_obs["ovT_rnd"][i],
                    data_obs["movT_rnd"][i],
                    ovTt[i],
                    movTt[i],
                    dovTt[i],
                    data_obs["e"][i],
                    data_obs["se"][i],
                    data_obs["mse"][i],
                    data_obs["se_rnd"][i],
                    data_obs["mse_rnd"][i],
                    dse[i],
                    Rse[i],
                    Rmse[i],
                    dRse[i],
                    data_obs["logL"][i],
                    data_obs["nI"][i],
                    data_obs["T_table"][i],
                    data_obs["sim"][i],
                    data_obs["fI"][i],
                ]
                for i, o in enumerate(data_obs["ov0"])
            ]
            if print_it:
                data_list_it = [
                    [
                        data_obs_it["N"],
                        data_obs_it["d"],
                        data_obs_it["lam"],
                        data_obs_it["n_sim"],
                        data_obs_it["n_iter"],
                        #data_obs_it["pseed"],
                        data_obs_it["iter"][i],
                        data_obs_it["# sources"][i],
                        data_obs_it["# obs"][i],
                        data_obs_it["init"][i],
                        data_obs_it["ov0"][i],
                        data_obs_it["mov0"][i],
                        dov_it[i],
                        data_obs_it["ov0_rnd"][i],
                        data_obs_it["mov0_rnd"][i],
                        ovt_it[i],
                        movt_it[i],
                        dovt_it[i],
                        data_obs_it["e"][i],
                        data_obs_it["logL"][i],
                        data_obs_it["sim"][i],
                        data_obs_it["fI"][i],
                        data_obs_it["converged"][i],
                    ]
                    for i, o in enumerate(data_obs_it["ov0"])
                ]
            o = "# obs"
            s = "# sources"

    s_N = r"$N$"
    s_d = r"$d$"
    s_l = r"$\lambda$"
    ov0 = r"$O_{t=0}$"
    mov0 = r"$MO_{t=0}$"
    ov0_rnd = r"$O_{t=0,RND}$"
    mov0_rnd = r"$MO_{t=0,RND}$"
    dov0 = r"$\delta O_{t=0}$"
    ov0t = r"$\widetilde{O}_{t=0}$"
    mov0t = r"$\widetilde{MO}_{t=0}$"
    dov0t = r"$\widetilde{\delta O}_{t=0}$"
    ovT = r"$O_{t=T}$"
    movT = r"$MO_{t=T}$"
    ovT_rnd = r"$O_{t=T,RND}$"
    movT_rnd = r"$MO_{t=T,RND}$"
    dovT = r"$\delta O_{t=T}$"
    ovTt = r"$\widetilde{O}_{t=T}$"
    movTt = r"$\widetilde{MO}_{t=T}$"
    dovTt = r"$\widetilde{\delta O}_{t=T}$"
    se_rnd = r"$SE_{RND}$"
    mse_rnd = r"$MSE_{RND}$"
    dse = r"$\delta SE$"
    Rse =  r"$\widetilde{SE}$"
    Rmse =  r"$\widetilde{MSE}$"
    dRse =  r"$\widetilde{\delta SE}$"
    fI = r"$f_I$"
    se = "SE"
    mse = "MSE"
    Delta = r"$\Delta$"

    data_frame = pd.DataFrame(
        data_list,
        columns=[
            s_N,
            s_d,
            s_l,
            Delta,
            "n_sim",
            "n_iter",
            #"pseed",
            s,
            o,
            "init",
            ov0,
            mov0,
            dov0,
            ov0_rnd,
            mov0_rnd,
            ov0t,
            mov0t,
            dov0t,
            ovT,
            movT,
            dovT,
            ovT_rnd,
            movT_rnd,
            ovTt,
            movTt,
            dovTt,
            "error",
            se,
            mse,
            se_rnd,
            mse_rnd,
            dse,
            Rse,
            Rmse,
            dRse,
            "logLikelihood",
            "# iter",
            "T",
            "sim",
            fI
        ],
    )
    data_frame[s_N] = data_frame[s_N].astype(int)
    data_frame[s_d] = data_frame[s_d].astype(int)
    data_frame[s_l] = data_frame[s_l].astype(float)
    data_frame[Delta] = data_frame[Delta].astype(int)
    data_frame["n_sim"] = data_frame["n_sim"].astype(int)
    data_frame["n_iter"] = data_frame["n_iter"].astype(int)
    #data_frame["pseed"] = data_frame["pseed"].astype(float)
    data_frame["init"] = data_frame["init"].astype(str)
    data_frame[ov0] = data_frame[ov0].astype(float)
    data_frame[mov0] = data_frame[mov0].astype(float)
    data_frame[dov0] = data_frame[dov0].astype(float)
    data_frame[ov0_rnd] = data_frame[ov0_rnd].astype(float)
    data_frame[mov0_rnd] = data_frame[mov0_rnd].astype(float)
    data_frame[ov0t] = data_frame[ov0t].astype(float)
    data_frame[mov0t] = data_frame[mov0t].astype(float)
    data_frame[dov0t] = data_frame[dov0t].astype(float)
    data_frame[ovT] = data_frame[ovT].astype(float)
    data_frame[movT] = data_frame[movT].astype(float)
    data_frame[dovT] = data_frame[dovT].astype(float)
    data_frame[ovT_rnd] = data_frame[ovT_rnd].astype(float)
    data_frame[movT_rnd] = data_frame[movT_rnd].astype(float)
    data_frame[ovTt] = data_frame[ovTt].astype(float)
    data_frame[movTt] = data_frame[movTt].astype(float)
    data_frame[dovTt] = data_frame[dovTt].astype(float)
    data_frame["error"] = data_frame["error"].astype(float)
    data_frame[se] = data_frame["SE"].astype(float)
    data_frame[mse] = data_frame["MSE"].astype(float)
    data_frame[se_rnd] = data_frame[se_rnd].astype(float)
    data_frame[mse_rnd] = data_frame[mse_rnd].astype(float)
    data_frame[dse] = data_frame[dse].astype(float)
    data_frame[Rse] = data_frame[Rse].astype(float)
    data_frame[Rmse] = data_frame[Rmse].astype(float)
    data_frame[dRse] = data_frame[dRse].astype(float)
    data_frame["logLikelihood"] = data_frame["logLikelihood"].astype(float)
    data_frame["# iter"] = data_frame["# iter"].astype(float)
    data_frame["T"] = data_frame["T"].astype(int)
    data_frame["sim"] = data_frame["sim"].astype(int)
    data_frame[fI] = data_frame[fI].astype(float)

    if print_it: #TO BE CHANGED!!!!
        data_frame_it = pd.DataFrame(
            data_list_it,
            columns=[
                s_N,
                s_d,
                s_l,
                "n_sim",
                "n_iter",
                #"pseed",
                "iter",
                s,
                o,
                "init",
                ov,
                mov,
                dov,
                ov_rnd,
                mov_rnd,
                ovt,
                movt,
                dovt,
                "error",
                "logLikelihood",
                "sim",
                fI,
                "converged",
            ],
        )
        data_frame_it[s_N] = data_frame_it[s_N].astype(int)
        data_frame_it[s_d] = data_frame_it[s_d].astype(int)
        data_frame_it[s_l] = data_frame_it[s_l].astype(float)
        data_frame_it["n_sim"] = data_frame_it["n_sim"].astype(int)
        data_frame_it["n_iter"] = data_frame_it["n_iter"].astype(int)
        #data_frame_it["pseed"] = data_frame_it["pseed"].astype(float)
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
        data_frame_it[fI] = data_frame_it[fI].astype(float)
        data_frame_it["converged"] = data_frame_it["converged"].astype(str)

    timestr = time.strftime("%Y%m%d-%H%M%S")# + "_" + str(random.randint(1,1000))
    if snap:
        data_frame[o] = data_frame[o].astype(float)
        if print_it:
            data_frame_it[o] = data_frame_it[o].astype(float)
        data_frame[s] = data_frame[s].astype(float)
        if print_it:
            data_frame_it[s] = data_frame_it[s].astype(float)
        file_name = (
            "data_BPEpi_{}_N{}_d{}_deltaMax{:.4f}_lam{:.2f}_SNAP_seed{}_".format(
                graph, N, d, S, lam, seed
            )
            + timestr
            + ".xz"
        )    
    else:
        if flag_obs:
            if flag_sources:
                data_frame[o] = data_frame[o].astype(float)
                if print_it:
                    data_frame_it[o] = data_frame_it[o].astype(float)
                data_frame[s] = data_frame[s].astype(float)
                if print_it:
                    data_frame_it[s] = data_frame_it[s].astype(float)
                file_name = (
                    "data_BPEpi_{}_N{}_d{}_deltaMax{:.4f}_lam{:.2f}_rhoMax{:.3f}_seed{}_".format(
                        graph, N, d, S, lam, M, seed
                    )
                    + timestr
                    + ".xz"
                )
            else:
                data_frame[o] = data_frame[o].astype(float)
                if print_it:
                    data_frame_it[o] = data_frame_it[o].astype(float)
                data_frame[s] = data_frame[s].astype(int)
                if print_it:
                    data_frame_it[s] = data_frame_it[s].astype(int)
                file_name = (
                    "data_BPEpi_{}_N{}_d{}_nsMax{}_lam{:.2f}_rhoMax{:.3f}_seed{}_".format(
                        graph, N, d, S, lam, M, seed
                    )
                    + timestr
                    + ".xz"
                )
        else:
            if flag_sources:
                data_frame[o] = data_frame[o].astype(int)
                if print_it:
                    data_frame_it[o] = data_frame_it[o].astype(int)
                data_frame[s] = data_frame[s].astype(float)
                if print_it:
                    data_frame_it[s] = data_frame_it[s].astype(float)
                file_name = (
                    "data_BPEpi_{}_N{}_d{}_deltaMax{:.4f}_lam{:.2f}_nobsMax{}_seed{}_".format(
                        graph, N, d, S, lam, M, seed
                    )
                    + timestr
                    + ".xz"
                )

            else:
                data_frame[o] = data_frame[o].astype(int)
                if print_it:
                    data_frame_it[o] = data_frame_it[o].astype(int)
                data_frame[s] = data_frame[s].astype(int)
                if print_it:
                    data_frame_it[s] = data_frame_it[s].astype(int)
                file_name = (
                    "data_BPEpi_{}_N{}_d{}_nsMax{}_lam{:.2f}_nobsMax{}_seed{}_".format(
                        graph, N, d, S, lam, M, seed
                    )
                    + timestr
                    + ".xz"
                )

    if print_it:
        saveObj = (data_frame, data_frame_it, np.array(data_obs["marginal0"]))
    else:
        saveObj = (data_frame, np.array(data_obs["marginal0"]))

    with lzma.open(save_dir + file_name, "wb") as f:
        pickle.dump(saveObj, f)
