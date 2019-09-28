import torch as tr
import numpy as np
from torch.autograd import Variable

import torch.nn as nn
from PIL import Image

FDTYPE = tr.float32
DEVICE = 'cuda'

def pow_10(x,dtype=FDTYPE,device=DEVICE): 

	return tr.pow(tr.tensor(10., dtype=dtype, device =device),x)

def support_1d(fun, x):
    assert 1<=x.ndim<=2
    return fun(x) if x.ndim == 2 else fun(x[None,:])[0]

def get_grid(r, i, j, cond):


    grid = np.meshgrid(r,r)

    grid = np.stack(grid,2)
    grid = grid.reshape(-1,2)
    
    num_point = len(grid)
    grid_cond = np.tile(cond[None,:], [num_point, 1])
    
    grid_cond[:,i] = grid[:,0]
    grid_cond[:,j] = grid[:,1]
    return grid_cond
def make_grid_points(D,ngrid,lim):
    idx_i, idx_j = 0,1
    eval_grid = np.linspace(-lim,lim,ngrid)
    cond_values = np.zeros(D)
    epsilon = 1.5
    eval_points = get_grid(eval_grid, idx_i, idx_j, cond_values)
    return eval_points, eval_grid


def load(fn='', size=200, max_samples=None):
    # returns x,y of black pixels
    pic = np.array(Image.open(fn).resize((size,size)).convert('L'))
    y_inv,x = np.nonzero(pic)
    y = size-y_inv-1
    if max_samples and x.size > max_samples:
        ixsel = np.random.choice(x.size, max_samples, replace=False)
        x, y = x[ixsel], y[ixsel]
    return x,y



