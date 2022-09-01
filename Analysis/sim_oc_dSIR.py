#import pandas as pd
import numpy as np
import networkx as nx
import random
import pandas as pd

from gen import *
from bpepi.Modules.st import *
from bpepi.Modules.fg import *
from XZtoDF import data_to_dict

import sys
import time
import lzma
import pickle
from pathlib import Path
import argparse
import warnings

def BPloop(
    f,
    list_obs,
    n_iter,
    tol,
    print_it,
    iter_space,
    tol2,
    it_max
):
    #Initialization
    for it in range(n_iter):
        e0 = f.iterate()
        if e0 < tol:
            break
    if e0 > tol:
        warnings.warn("Warning... Initialization is not converging")
    f.reset_obs(list_obs)
    e = np.nan
    #BP iteration
    tol2 = 1e-2
    it_max = 10000
    if print_it:
        marg_list=[f.marginals()]
        it_list=[0]
        e_list=[e]
        logL_list = [f.loglikelihood()]
    for it in np.arange(1,n_iter+1):
        e = f.iterate()
        if print_it and (
            (it % iter_space == 0) or (e < tol) or (it == n_iter)
        ):
            marg_list.append(f.marginals())
            it_list.append(it)
            e_list.append(e)
            logL_list.append(f.loglikelihood())
        if e < tol:
            break
    while (e > tol) and (e < tol2):
        it = it + 1
        e = f.iterate()
        if print_it and ((it % iter_space == 0) or (e < tol) or (it == it_max)):
            marg_list.append(f.marginals())
            it_list.append(it)
            e_list.append(e)
            logL_list.append(f.loglikelihood())
        if it == it_max:
            break
    
    if not print_it : 
        marg_list = [f.marginals()]
        it_list = [it]
        e_list = [e]
        logL_list = [f.loglikelihood()]

    return marg_list, e_list, it_list, logL_list

def main():
    parser = argparse.ArgumentParser(description="Compute marginals using BPEpI")
    parser.add_argument(
        "--save_dir",
        type=str,
        default="../data/check/",
        help="saving directory for data",
    )
    parser.add_argument(
        "--save_DF_dir",
        type=str,
        default="../data/check/",
        help="saving directory for data frames",
    )
    parser.add_argument(
        "--file_name",
        type=str,
        default="data",
        help="name of the file in which the marginals will be saved",
    )
    parser.add_argument(
        "--graph", type=str, default="rrg", help="Type of random graph"
    )
    parser.add_argument(
        "--N", type=int, default=10000, help="Number of individuals", nargs="+"
    )
    parser.add_argument(
        "--d", type=int, default=3, help="degree of RRG", nargs="+"
    )
    parser.add_argument(
        "--lam", type=float, default=1.0, help="lambda", nargs="+"
    )
    group_s = parser.add_mutually_exclusive_group(required=True)
    group_s.add_argument(
        "--n_sources",
        type=int,
        default=1,
        help="number of sources, pass as multiple arguments, e.g. 2 4 8",
        nargs="+",
    )
    group_s.add_argument(
        "--delta",
        type=float,
        default=0.01,
        help="fraction of sources, pass as multiple arguments, e.g. 0.1 0.3 0.5",
        nargs="+",
    )
    group_o = parser.add_mutually_exclusive_group(required=True)
    group_o.add_argument(
        "--n_obs",
        type=int,
        default=[-1],
        help="number of observations, pass as multiple arguments, e.g. 2 4 8",
        nargs="+",
    )
    group_o.add_argument(
        "--rho",
        type=float,
        default=[-1],
        help="fraction of observations, pass as multiple arguments, e.g. 0.1 0.3 0.5",
        nargs="+",
    )
    parser.add_argument(
        "--nsim", type=int, default=25, help="number of simulations"
    )
    parser.add_argument(
        "--print_it",
        action="store_true",
        help="If false, save just the marginals at convergence. If true, save marginals every 'iter_space' iterations of BP",
    )
    parser.add_argument(
        "--iter_space",
        type=int,
        default=100,
        help="Space between saved iterations",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=0,
        help="seed for the number generators",
    )
    parser.add_argument(
        "--tol", type=float, default=1e-9, help="Tolerance of the difference between marginals to stop BP "
    )
    parser.add_argument(
        "--n_iter",
        type=int,
        default=2000,
        help="Max number of iterations of the algorithm",
    )
    group_ot = parser.add_mutually_exclusive_group(required=True)
    group_ot.add_argument(
        "--sens",
        action="store_const",
        const='sensors',
        dest="obs_type",
        help="Snapshot observation",
    )
    group_ot.add_argument(
        "--snap",
        action="store_const",
        const='snapshot',
        dest="obs_type",
        help="Snapshot observation",
    )
    parser.add_argument(
        "--snap_time",
        type=int,
        default=-1,
        help="Time at which taking the snapshot. Random by default",
    )
    parser.add_argument(
        "--T_max",
        type=int,
        default=100,
        help="Max number of timesteps of the simulation",
    )
    group_d = parser.add_mutually_exclusive_group(required=True)
    group_d.add_argument(
        "--Delta",
        type=int,
        default=-1,
        help="Lenght of mask array, filled with ones",
    )
    group_d.add_argument(
        "--mu",
        type=float,
        default=-1,
        help="Value of recovery probability of the SIR model to simulate through dSIR",
    )
    group_d.add_argument(
        "--mask",
        type=float,
        default=1.,
        help="Mask array",
        nargs="+",
    )
    group_d.add_argument(
        "--SI",
        action="store_true",
        help="Set the mask array in order to simulate SI model",
    )
    parser.add_argument(
        "--tol2", type=float, default=1e-2, help="Tolerance of the difference between marginals to stop BP for the second part "
    )
    parser.add_argument(
        "--it_max", type=int, default=10000, help="Max number of iterations of the algorithm, after the first threshold "
    )
    parser.add_argument(
        "--save_marginals",
        action="store_true",
        help="If false, save just the Data Frame. If true, save also the marginals found by BP",
    )
    parser.add_argument(
        "--SIR_sim",
        action="store_true",
        help="If true, simulates a conventional SIR model",
    )

    args = parser.parse_args()
    print("arguments:")
    print(args)

    if args.n_sources == 0:
        warnings.warn("YOU CANNOT HAVE ZERO SOURCES!")
        sys.exit()
    save_dir = args.save_dir
    save_DF_dir = args.save_DF_dir
    path_save = Path(save_dir)
    if not path_save.exists():
        warnings.warn("SAVING FOLDER DOES NOT EXIST")
    graph = args.graph
    if graph == "rrg":
        def generate_graph(N, d):
            return nx.random_regular_graph(n=N, d=d)
    elif graph == "tree":
        def generate_graph(N, d):
            return nx.full_rary_tree(r=d, n=N)
    else:
        warnings.warn("GRAPH TYPE NOT ALLOWED")
    N_table = args.N
    d_table = args.d
    lam_table = args.lam
    if args.delta is not None:
        sources_table = args.delta
        s_type = "delta"
    else:
        sources_table = args.n_sources
        s_type = "n_sources"
    if args.rho is not None:
        obs_table = args.rho
        o_type = "rho"
    else:
        obs_table = args.n_obs
        o_type = "n_obs"
    n_sim = args.nsim
    print_it = args.print_it
    iter_space = args.iter_space
    seed = args.seed  # setting seed everywhere for reproducibility TBD
    random.seed(seed)
    np.random.seed(seed)
    tol = args.tol
    n_iter = args.n_iter
    T_max=args.T_max
    mu=0
    if args.SI == True:
        mask = ["SI"]
        Delta = T_max + 1
        mask_type = "SI"
    elif args.Delta != -1:
        mask=[1]*args.Delta
        Delta = args.Delta
        mask_type = "dSIR_one"
    elif args.mu != -1:
        mu = args.mu
        mask = [(1-mu)**i for i in range(T_max+1)]
        Delta = T_max + 1
        mask_type = "dSIR_exp"
    else:
        mask = args.mask
        Delta = len(mask)
        mask_type = "dSIR_custom"
    tol2 = args.tol2
    it_max = args.it_max
    save_marginals = args.save_marginals
    SIR_sim = args.SIR_sim

    dict_list = []
    t1 = time.time()
    t2 = time.time()
    for i_N, N in enumerate(N_table):
        for i_d, d in enumerate(d_table):
            for i_l, lam in enumerate(lam_table):
                for i_S, S in enumerate(sources_table):
                    for i_M, M in enumerate(obs_table):
                        for sim in range(n_sim):
                            if args.delta is None:
                                pseed=S/N
                            else: pseed=S
                            G = generate_graph(N=N, d=d)
                            for (u,v) in G.edges():
                                G.edges[u,v]['lambda'] = lam
                            if SIR_sim == True: 
                                mask_type = "SIR"
                                ground_truth = simulate_one_SIR(G, s_type=s_type, S = S, mu=mu, T_max=T_max)
                            else: ground_truth = simulate_one_detSIR(G, s_type=s_type, S = S, mask = mask, T_max=T_max)
                            if len(ground_truth) > T_max : 
                                warnings.warn("The simulation exeeds the maximum time limit!")
                                sys.exit()
                            T = len(ground_truth) - 1
                            contacts = generate_contacts(G, T, lam)
                            if args.obs_type == "sensors":
                                list_obs = generate_sensors_obs(ground_truth, o_type=o_type, M=M)
                                list_obs_all = generate_sensors_obs(ground_truth, o_type="rho", M=1)
                                fS = np.mean(ground_truth[-1]==0)
                                fI = np.mean(ground_truth[-1]==1)
                                TO=T
                            else:
                                list_obs, fS, fI, TO = generate_snapshot_obs(ground_truth, o_type=o_type, M=M, snap_time=args.snap_time)
                                list_obs_all = generate_sensors_obs(ground_truth, o_type="rho", M=1)

                            f_rnd = FactorGraph(
                                N=N, T=T, contacts=contacts, obs=[], delta=pseed, mask=mask, mask_type=mask_type
                            )
                            f_informed = FactorGraph(
                                N=N, T=T, contacts=contacts, obs=list_obs_all, delta=pseed, mask=mask, mask_type=mask_type
                            )
                            marg_list_rnd, eR_list, itR_list, logLR_list = BPloop(
                                f_rnd,
                                list_obs,
                                n_iter,
                                tol,
                                print_it,
                                iter_space,
                                tol2,
                                it_max
                            )
                            marg_list_inf, eI_list, itI_list, logLI_list = BPloop(
                                f_informed,
                                list_obs,
                                n_iter,
                                tol,
                                print_it,
                                iter_space,
                                tol2,
                                it_max
                            )
                            print(
                                f"\r N: {i_N+1}/{len(N_table)} - d: {i_d+1}/{len(d_table)} - lam: {i_l+1}/{len(lam_table)} - S: {i_S+1}/{len(sources_table)} - M: {i_M+1}/{len(obs_table)} - sim: {sim+1}/{n_sim} - time = {time.time()-t2:.2f} s - total time = {time.time()-t1:.0f} s"
                            )
                            t2 = time.time()
                            timestr = "_" +time.strftime("%Y%m%d-%H%M%S") + "_" + str(random.randint(1,1000))
                            if print_it : file_name = "IT_" + args.file_name + timestr + ".xz"  
                            else : file_name = args.file_name + timestr + ".xz"  

                            saveObj = (
                                graph, 
                                N,
                                d,
                                lam,
                                s_type,
                                S,
                                o_type,
                                M,
                                iter_space,
                                seed,
                                tol,
                                n_iter,
                                args.obs_type,
                                args.snap_time,
                                T_max,
                                mask,
                                mask_type,
                                tol2,
                                it_max,
                                ground_truth,
                                marg_list_rnd, 
                                marg_list_inf,
                                G,
                                T,
                                list_obs,
                                fS,
                                fI,
                                TO,
                                Delta,
                                eR_list,
                                eI_list,
                                itR_list,
                                itI_list,
                                logLR_list,
                                logLI_list
                            )
                            if save_marginals :
                                with lzma.open(save_dir + file_name, "wb") as f:
                                    pickle.dump(saveObj, f)
                            dict_list = dict_list + data_to_dict(saveObj)
    data_frame = pd.DataFrame(dict_list)
    timestr = time.strftime("%Y%m%d-%H%M%S") + "_" + str(random.randint(1,1000))
    if print_it : file_name = "DF_IT_" + timestr + ".xz"  
    else : file_name = "DF_" + timestr + ".xz" 
    with lzma.open(save_DF_dir + file_name, "wb") as f:
        pickle.dump(data_frame, f)

if __name__ == "__main__":
    main()