import sys
import pickle
sys.path.append('/.../bpepi/bpepi')
from st import SparseTensor
from fg import FactorGraph

def test():
    #import data
    with open('/.../bpepi/tests/test_data/contacts', 'rb') as f:
        contacts = pickle.load(f)
    with open('/.../bpepi/tests/test_data/list_obs', 'rb') as f:
        list_obs = pickle.load(f)
    with open('/.../bpepi/tests/test_data/params', 'rb') as f:
        params = pickle.load(f)
    sib_marginals = np.load('/.../bpepi/tests/test_data/sib_marginals.npz')['arr_0']
    
    code_fg = FactorGraph(params[0], params[1], contacts, list_obs, params[2])
    code_fg.update()
    code_marginals = code_fg.marginals()
    
    if np.allclose(np.sum(code_marginals - sib_marginals, axis=1), np.zeros(params[0]), atol=1e-14) == True:
        return print('passed')
    else:
        return print('failed')