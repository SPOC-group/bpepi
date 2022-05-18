import sys
import os
import pickle
import numpy as np
sys.path.insert(0, os.path.abspath(os.path.join(os.path.join(os.path.dirname(__file__),".."), "bpepi")))
from st import SparseTensor
from fg import FactorGraph

path = os.path.abspath(os.path.join(os.path.dirname(__file__), "test_data"))

def test():
    passed = []
    for i in range(1, 3):
        path = os.path.abspath(os.path.join(os.path.dirname(__file__), 'test_data/set_'+str(i)))
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
    return print(sum(passed), 'tests passed out of', len(passed))

test()
