import torch as tr
import numpy as np
from torch.autograd import Variable

import torch.nn as nn
from PIL import Image
import networkx as nx
import math
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



def generate_graph(N, completeness):

	G = nx.Graph()
	Npair = int(N*(N-1)/2)
	I = np.zeros((2, Npair))

	# generate all pairwise edges
	k=0
	for i in range(N):
		for j in range(i,N):
			if(i!=j):
				I[:,k]=[i,j]
				k=k+1

	# now keep a portion of the edges
	e = math.ceil(completeness*Npair)
	ind = np.random.choice(Npair, e,  replace=False)
	I = I[:, ind]
	vals = np.ones(e, dtype=np.double)
	rows = (np.asarray(I[0,:])).astype(int)
	cols = (np.asarray(I[1,:])).astype(int)

	G = np.zeros((N,N))
	G[rows, cols] = vals
	
	e = I.shape[1]
	edges = I
	return edges

def quaternion_prod(a,b):
	shape_a = a.shape
	c = tr.zeros_like(a)
	c[:,:,1:] =  tr.cross(a[:,:,1:],b[:,:,1:]) +a[:,:,0].unsqueeze(-1)*b[:,:,1:] + b[:,:,0].unsqueeze(-1)*a[:,:,1:]
	c[:,:,0] = a[:,:,0]*b[:,:,0] -  tr.einsum('ijd,ijd->ij',a[:,:,1:],b[:,:,1:]) 
	return c

def quaternion_exp_map(a,v):
	alpha  = tr.norm(v,dim=-1)
	beta = tr.sin(alpha)/alpha
	return tr.einsum('...d,...->...d' ,a,tr.cos(alpha)) + tr.einsum('...d,...->...d' ,v,beta)

def quaternion_proj(q,g):
	prod = tr.einsum('...d,...d->...',q,g)
	return g - tr.einsum('...d,...->...d',q,prod)

def quaternion_geodesic_distance(a,b):
	return 2*tr.acos(tr.abs(tr.einsum('...i,...i->...',a,b)))







