import sys
import os
import pickle
import numpy as np
import networkx as nx
import time
import argparse
sys.path.insert(0, os.path.abspath(os.path.join(os.path.join(os.path.dirname(__file__),".."), "bpepi")))
from st import SparseTensor
from fg import FactorGraph
from sim_on_cluster import generate_contacts, generate_one_conf, generate_obs

parser = argparse.ArgumentParser(description="parameters")
parser.add_argument("--type", type=str, default='t', dest='type', help='type of test')
#parameters for type = i
parser.add_argument("--max_it", type=int, default='200', dest='max_it', help='max number of iterations in BP update')
#parameters for type = s
parser.add_argument("--N", type=int, default='5000', dest='N', help='# of nodes')
parser.add_argument("--d", type=int, default='3', dest='d', help='degree of graph')
parser.add_argument("--T", type=int, default='10', dest='T', help='final time')
parser.add_argument("--lam", type=float, default='1', dest='lam', help='transmission probability')
parser.add_argument("--delta", type=float, default='0.01', dest='delta', help='probability of being source')
parser.add_argument("--rho", type=float, default='0.1', dest='rho', help='fraction of observations')

args = parser.parse_args()

typ = args.type
max_it = args.max_it
N = args.N
d = args.d
T = args.T
lam = args.lam
delta = args.delta
rho = args.rho

def test(type='t'):
    """Function to check agreement with SIB

    Args:
    type (str): Which test to perform. Current options are {'t', 'i'}. 't' checks if marginals agree on tree graphs. 'i' checks if marginals from informed and random initializations in bpepi agree in regimes where they do not in SIB.
    """ 
    passed = []
    if type=='t':
        num_dir = 0
        count_path = os.path.abspath(os.path.join(os.path.dirname(__file__), 'test_data/tree'))
        for root, dirs, files in os.walk(count_path):
            for name in dirs:
                num_dir += 1
        for i in range(0, num_dir):
            path = os.path.join(count_path, str(i))
            with open(path+'/contacts', 'rb') as f:
                contacts = pickle.load(f)
            with open(path+'/list_obs', 'rb') as f:
                list_obs = pickle.load(f)
            with open(path+'/params', 'rb') as f:
                params = pickle.load(f)
            sib_marginals = np.load(path+'/sib_marginals.npz')['arr_0']
            code_fg = FactorGraph(params[0], params[1], contacts, list_obs, params[2])
            code_fg.update()
            code_marginals = code_fg.marginals()
            if np.allclose(np.sum(code_marginals - sib_marginals, axis=1), np.zeros(params[0]), atol=1e-14) == True:
                passed.append(1)
            else:
                passed.append(0)
        print(sum(passed), 'tests passed out of', len(passed))
    if type=='i':
        save_path = os.path.abspath(os.path.join(os.path.dirname(__file__), "."))
        num_dir=0
        count_path = os.path.abspath(os.path.join(os.path.dirname(__file__), 'test_data/initialization'))
        for root, dirs, files in os.walk(count_path):
            for name in dirs:
                print(name)
                num_dir += 1
        print(num_dir)
        for i in range(0, num_dir):
            path = os.path.join(count_path, str(i))
            with open(path+'/contacts', 'rb') as f:
                contacts = pickle.load(f)
            with open(path+'/list_obs', 'rb') as f:
                list_obs = pickle.load(f)
            with open(path+'/list_obs_all', 'rb') as f:
                list_obs_all = pickle.load(f)
            with open(path+'/params', 'rb') as f:
                params = pickle.load(f)
            sib_marginals_rnd = np.load(path+'/sib_marginals_rnd.npz')['arr_0']
            sib_marginals_inf = np.load(path+'/sib_marginals_inf.npz')['arr_0']
            code_f_rnd = FactorGraph(params[0], params[1], contacts, [], params[2])
            code_f_rnd.update(maxit=max_it, tol=1e-9)
            code_f_rnd.reset_obs(list_obs)
            code_f_rnd.update(maxit=max_it, tol=1e-9)
            code_marginals_rnd = code_f_rnd.marginals()
            np.savez(save_path+'/bpepi_marginals_rnd.npz', code_marginals_rnd)
            code_f_inf = FactorGraph(params[0], params[1], contacts, list_obs_all, params[2])
            code_f_inf.update(maxit=max_it, tol=1e-9)
            code_f_inf.reset_obs(list_obs)
            code_f_inf.update(maxit=max_it, tol=1e-9)
            code_marginals_inf = code_f_inf.marginals()
            np.savez(save_path+'/bpepi_marginals_inf.npz', code_marginals_inf)
            tolerances = [1e-12, 1e-11, 1e-10, 1e-9, 1e-8, 1e-7, 1e-6, 1e-5, 1e-4, 1e-3, 1e-2, 1e-1, 1]
            distance_1 = '> 1'
            for tol in tolerances:
                if np.allclose(np.sum(np.abs(code_marginals_rnd - sib_marginals_rnd), axis=1), np.zeros(params[0]), atol=tol):
                    distance_1 = '<'+str(tol)
                    break
            distance_2 = '> 1'
            for tol in tolerances:
                if np.allclose(np.sum(np.abs(code_marginals_inf - sib_marginals_rnd), axis=1), np.zeros(params[0]), atol=tol):
                    distance_2 = '<'+str(tol)
                    break
            distance_3 = '> 1'
            for tol in tolerances:
                if np.allclose(np.sum(np.abs(sib_marginals_rnd - sib_marginals_inf), axis=1), np.zeros(params[0]), atol=tol):
                    distance_3 = '<'+str(tol)
                    break
        print('bpepi rnd marginals and sib rnd marginals distance', distance_1)
        print('bpepi inf marginals and sib rnd marginals distance', distance_2)
        print('sib inf marginals and sib rnd marginals distance', distance_3)
    if type=='s':
        g = nx.random_regular_graph(d=d,n=N)
        contacts = generate_contacts(g, T, lam)
        status_nodes = generate_one_conf(g, lamb=lam, T=T+1, percentage_infected=delta)
        list_obs = generate_obs(status_nodes, frac_obs = rho)
        list_obs_all = generate_obs(status_nodes, frac_obs = 1.)
        test = FactorGraph(N, T, contacts, list_obs, delta)
        times = []
        for i in range(5):
            t1 = time.time()
            test.iterate()
            t2 = time.time()
            times.append(t2-t1)
        print('average time taken for single BP iteration:', np.mean(np.array(times)))

test(type=typ)
