import pandas as pd
import numpy as np
import random
import argparse

from measures import *

import sys
import os
import time
import lzma
import pickle

def main():
    parser = argparse.ArgumentParser(description="Compute DataFrame of observables from data files")
    parser.add_argument(
        "--load_dir",
        type=str,
        default="../data/check/",
        help="load directory",
    )
    parser.add_argument(
        "--save_dir",
        type=str,
        default="../data/check/",
        help="saving directory",
    )
    parser.add_argument(
        "--print_it",
        action="store_true",
        help="If false, load just the marginals at convergence. If true, load also iteration-marginals",
    )
    parser.add_argument(
        "--init_DF",
        action="store_true",
        help="If false, creates Data Frame just from the XZ data. If true, load also the Data Frames contained in load_dir",
    )
    args = parser.parse_args()
    load_dir=args.load_dir
    save_dir = args.save_dir
    print_it = args.print_it

    if args.init_DF == True:
        data_frame = None
        for filename in os.listdir(load_dir):
            path = os.path.join(load_dir, filename)
            if not os.path.isdir(path):
                if ( (print_it == True) and (filename.startswith( "DF_IT_" ))) or ( (print_it == False) and (filename.startswith( "DF_" )) and not (filename.startswith( "DF_IT_" )) ):
                    with lzma.open(load_dir + filename, "rb") as f:
                        if data_frame == None : data_frame = pickle.load(f)
                        else: data_frame = pd.concat([data_frame,pickle.load(f)],ignore_index=True)
    else: data_frame = None

    for filename in os.listdir(load_dir):
        path = os.path.join(load_dir, filename)
        if not os.path.isdir(path):
            if not filename.startswith('.'):
                if ( (print_it == True) and (filename.startswith('IT_'))) or ( (print_it == False) and not (filename.startswith('DF_')) ):
                    with lzma.open(load_dir + filename, "rb") as f:
                        data = pickle.load(f)
                        [graph, 
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
                        obs_type,
                        snap_time,
                        T_max,
                        mask,
                        #mask_type,
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
                        #Delta,
                        #eR_list,
                        #eI_list,
                        #itR_list,
                        #itI_list,
                        #logLR_list,
                        #logLI_list
                        ] = data
                        Delta = T_max + 1
                        mask_type = "SI"
                        B_tab = [marg_list_rnd, marg_list_inf]
                        init_tab = ["rnd","inf"]
                        #e_tab = [eR_list, eI_list]
                        #it_tab = [itR_list, itI_list]
                        #logL_tab = [logLR_list, logLI_list]
                        for i, B_list in enumerate(B_tab):
                            init = init_tab[i]
                            for it_idx, B in enumerate(B_list):                        
                                MI0 = B[:, 0]
                                MS0 = 1-MI0
                                MR0 = np.zeros_like(MS0)
                                M0 = np.array([ MS0, MI0, MR0])
                                #Compute general marginals:
                                pS = 1 - np.cumsum(B,axis=1)
                                pI = np.array([ np.array([ sum(b[1+t-Delta:1+t]) if t>=Delta else sum(b[:t+1]) for t in range(T+2)]) for b in B])
                                pR = 1 - pS - pI
                                #Take last time
                                ###Ms0 = np.cumsum(Bs,axis=1)[:,T]
                                MST = pS[:,T]
                                MIT = pI[:,T]
                                MRT = pR[:,T]
                                MT = np.array([ MST, MIT, MRT])
                                x0_inf = np.argmax(M0,axis=0)
                                xT_inf = np.argmax(MT,axis=0)
                                ti_str = ti_star(ground_truth)

                                ov0 = OV(ground_truth[0], x0_inf)
                                ov0_rnd = OV_rnd(ground_truth[0], M0)
                                mov0 = MOV(M0)
                                mov0_rnd = MOV_rnd(M0)
                                ovT = OV(ground_truth[T], xT_inf)
                                ovT_rnd = OV_rnd(ground_truth[T], MT)
                                movT = MOV(MT)
                                movT_rnd = MOV_rnd(MT)
                                ti_inf = ti_inferred(B)
                                ti_rnd = ti_random(B)
                                se = SE(ti_str, ti_inf)
                                mse = MSE(B, ti_inf)
                                se_rnd = SE(ti_str, ti_rnd)
                                mse_rnd = MSE(B, ti_rnd)
                                e = np.NaN#e_tab[i][it_idx]
                                it = np.NaN#it_tab[i][it_idx]
                                it_final = np.NaN#it_tab[i][-1]
                                logL = np.NaN#logL_tab[i][it_idx]
                                ov0t = (ov0 - ov0_rnd) / (1 - ov0_rnd)
                                mov0t = (mov0 - mov0_rnd) / (1 - mov0_rnd)
                                ovTt = (ovT - ovT_rnd) / (1 + 1e-12 - ovT_rnd)
                                movTt = (movT - movT_rnd) / (1 + 1e-12 - movT_rnd)
                                Rse = (se_rnd - se) / se_rnd
                                Rmse = (mse_rnd - mse) / mse_rnd
                                keys = [
                                    "init",
                                    "graph_type",
                                    "$N$",
                                    "$d$",
                                    "$\lambda$",
                                    "s_type",
                                    "S",
                                    "o_type",
                                    "M",
                                    "iter_space",
                                    "seed",
                                    "tol",
                                    "n_iter",
                                    "obs_type",
                                    "snap_time",
                                    "T_max",
                                    "mask_type",
                                    "tol2",
                                    "it_max",
                                    "$T$",
                                    "$f_S$",
                                    "$f_I$",
                                    "$T_O$",
                                    "$\Delta$",
                                    "error",
                                    "iteration",
                                    "it_final"
                                    "logL",
                                    "$\text{O}_{t=0}$",
                                    "$\text{O}_{t=0,RND}$",
                                    "$\text{MO}_{t=0}$",
                                    "$\text{MO}_{t=0,RND}$",
                                    "$\delta \text{O}_{t=0}$",
                                    "$\widetilde{\text{O}}_{t=0}$",
                                    "$\widetilde{\text{MO}}_{t=0}$",
                                    "$\widetilde{\delta \text{O}}_{t=0}$",
                                    "$\text{O}_{t=T}$",
                                    "$\text{O}_{t=T,RND}$",
                                    "$\text{MO}_{t=T}$",
                                    "$\text{MO}_{t=T,RND}$",
                                    "$\delta \text{O}_{t=T}$",
                                    "$\widetilde{\text{O}}_{t=T}$",
                                    "$\widetilde{\text{MO}}_{t=T}$",
                                    "$\widetilde{\delta \text{O}}_{t=T}$",
                                    "SE",
                                    "MSE"
                                    "$\text{SE}_{RND}$",
                                    "$\text{MSE}_{RND}$",
                                    "$\delta \text{SE}$",
                                    "$R_{\text{SE}}$",
                                    "$R_{\text{MSE}}$",
                                    "$\delta R_{\text{SE}}$",
                                ]
                                values = [
                                    init,
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
                                    obs_type,
                                    snap_time,
                                    T_max,
                                    mask_type,
                                    tol2,
                                    it_max,
                                    T,
                                    fS,
                                    fI,
                                    TO,
                                    Delta,
                                    e,
                                    it,
                                    it_final,
                                    logL,
                                    ov0,
                                    ov0_rnd,
                                    mov0,
                                    mov0_rnd,
                                    ov0 - mov0,
                                    ov0t,
                                    mov0t,
                                    ov0t - mov0t,
                                    ovT,
                                    ovT_rnd,
                                    movT,
                                    movT_rnd,
                                    ovT - movT,
                                    ovTt,
                                    movTt,
                                    ovTt - movTt,
                                    se,
                                    mse,
                                    se_rnd,
                                    mse_rnd,
                                    se - mse,
                                    Rse,
                                    Rmse,
                                    Rse - Rmse
                                ]
                                data_dict = dict(zip(keys, values))
                                    
                                if data_frame is None: data_frame = pd.DataFrame(data_dict, index=[0])
                                else : data_frame = pd.concat([data_frame,pd.DataFrame(data_dict, index=[0])],ignore_index=True)
    timestr = time.strftime("%Y%m%d-%H%M%S") + "_" + str(random.randint(1,1000))
    if print_it : file_name = "DF_IT_" + timestr + ".xz"  
    else : file_name = "DF_" + timestr + ".xz" 
    with lzma.open(save_dir + file_name, "wb") as f:
        pickle.dump(data_frame, f)

if __name__ == "__main__":
    main()

