import torch as tr
import numpy as np
from torch.autograd import Variable

import torch.nn as nn
from PIL import Image
import networkx as nx
import math
from torch import autograd
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

def quaternion_exp_map(a,v, north_hemisphere=False):
	alpha  = tr.norm(v[:,:,1:],dim=-1)
	tmp = tr.sin(alpha)/alpha
	beta = tr.ones_like(alpha)
	mask = (alpha>0.)
	beta[mask] = tmp[mask]

	#tmp = tr.einsum('...d,...->...d' ,a,tr.cos(alpha)) + tr.einsum('...d,...->...d' ,v,beta)
	exp_v = tr.ones_like(v)
	exp_v[:,:,0] = tr.cos(alpha)
	exp_v[:,:,1:] = tr.einsum('...d,...->...d' ,v[:,:,1:],beta)
	prod = quaternion_prod(a, exp_v)
	prod[:,:,0] = prod[:,:,0].clamp(0.)
	prod = prod/tr.norm(prod,dim=-1).unsqueeze(-1)
	return prod

	# if north_hemisphere:
	# 	tmp[:,:,0] = tmp[:,:,0].clamp(0.)
	# 	tmp= tmp/tr.norm(tmp,dim=-1).unsqueeze(-1)
	# return tmp

def sphere_exp_map(a,v, north_hemisphere=False):
	alpha  = tr.norm(v,dim=-1)
	tmp = tr.sin(alpha)/alpha
	beta = tr.ones_like(alpha)
	mask = (alpha>0.)
	beta[mask] = tmp[mask]

	tmp = tr.einsum('...d,...->...d' ,a,tr.cos(alpha)) + tr.einsum('...d,...->...d' ,v,beta)

	return tmp

def quaternion_proj(qq,g):
	prod = tr.einsum('...d,...d->...',qq,g)
	return g - tr.einsum('...d,...->...d',qq,prod)

def sphere_proj(qq,g):
	prod = tr.einsum('...d,...d->...',qq,g)
	return g - tr.einsum('...d,...->...d',qq,prod)


# def quaternion_geodesic_distance(a,b):
# 	return 2*tr.acos(tr.abs(tr.einsum('...i,...i->...',a,b)))



class Stable_squared_angle(tr.autograd.Function):


	@staticmethod
	def forward(ctx, X):
		ctx.save_for_backward(X)
		with  tr.enable_grad():
			return tr.acos(2*X**2-1)**2
		


	@staticmethod
	def backward(ctx, grad_output):
		X, = ctx.saved_tensors
		Y = tr.sqrt(1-X**2)

		mask = (Y>0.)
		ratio = tr.asin(Y)/Y
		tmp = tr.ones_like(X)
		tmp[mask] = ratio[mask]
		gradients = -8*tr.sign(X)*tmp*grad_output


		#return ss
		return gradients


stable_squared_angle = Stable_squared_angle.apply



class StableIdentity(tr.autograd.Function):
	@staticmethod
	def forward(ctx, X):
				
		#ctx.save_for_backward(X)
		return X

	@staticmethod
	def backward(ctx, grad_output):
		mask = tr.isfinite(grad_output)
		grad = tr.zeros_like(grad_output)
		grad[mask] = grad_output[mask]
		return grad

stableIdentity = StableIdentity.apply


class Quaternion_geodesic_distance(tr.autograd.Function):
	# takes a input two tensors   ...Md and ...Md

	@staticmethod
	def forward(ctx, X,Y):
		
		with  tr.enable_grad():
			#norm_X =  X/tr.norm(X,dim=-1).unsqueeze(-1)
			#norm_Y =  Y/tr.norm(Y,dim=-1).unsqueeze(-1)
			prod = tr.einsum('...ki,...li->...kl',X,Y)**2
			#loss = tr.acos(prod.clamp(min=-1.,max=1.))
			prodClamp = prod.clamp(max=1.)
			loss = 2*tr.atan( tr.sqrt(1-prodClamp)/prodClamp )
		ctx.save_for_backward(X,Y,loss)
		return loss

	@staticmethod
	def backward(ctx, grad_output):
		X,Y,_ = ctx.saved_tensors
		return grad_quaternion_geodesic_dist(X,Y,grad_output)
	# def backward(ctx, grad_output):
	# 	eps = 1e-15
	# 	X,Y = ctx.saved_tensors

	# 	w = _norm_im_a_inv_times_b(X,Y)

	# 	#ww = quaternion_a_inv_times_b(X,Y)
	# 	#w2 = tr.norm(ww,dim=-1)

	# 	mask  = (w>eps)
	# 	ratio = grad_output/w
	# 	weights = tr.zeros_like(grad_output)
	# 	weights[mask] = ratio[mask]
	# 	tmp_x = tr.einsum('nkl,nli->nki', weights,Y)
	# 	gradients_x = - _quaternion_a_inv_times_b(X, tmp_x, with_0_comp = False)
	# 	tmp_y = tr.einsum('nkl,nki->nli', weights,X)
	# 	gradients_y =  - _quaternion_a_inv_times_b(Y, tmp_y, with_0_comp = False)
	# 	#return aa
	# 	return gradients_x, gradients_y



	# def backward(ctx, grad_output):

	# 	X,Y,loss = ctx.saved_tensors
	# 	#if X.requires_grad and Y.requires_grad:
	# 	gradient =  autograd.grad(outputs=loss, inputs=[X,Y], grad_outputs=grad_output)
	# 	mask_1 = tr.isfinite(gradient[0])
	# 	mask_2 = tr.isfinite(gradient[1])
	# 	grad_x = tr.zeros_like(X)
	# 	grad_y = tr.zeros_like(Y)
	# 	grad_x[mask_1] = gradient[0][mask_1]
	# 	grad_y[mask_2] = gradient[1][mask_2]
	# 	# elif not X.requires_grad:
	# 	# 	gradient =  autograd.grad(outputs=loss, inputs=Y, grad_outputs=grad_output)
	# 	# 	mask_2 = tr.isfinite(gradient[0])
	# 	# 	grad_x = None	
	# 	# 	grad_y = tr.zeros_like(Y)
	# 	# 	grad_y[mask_2] = gradient[1][mask_2]
	# 	# elif not Y.requires_grad:
	# 	# 	gradient =  autograd.grad(outputs=loss, inputs=X, grad_outputs=grad_output)
	# 	# 	mask_1 = tr.isfinite(gradient[0])
	# 	# 	grad_x = tr.zeros_like(X)	
	# 	# 	grad_y = None
	# 	# 	grad_y[mask_1] = gradient[0][mask_1]
	# 	return grad_x, grad_y

		#return grad_quaternion_geodesic_dist(X,Y,grad_output)



quaternion_geodesic_distance = Quaternion_geodesic_distance.apply

# def quaternion_geodesic_distance(X,Y):
# 	#prod = 2*tr.einsum('...ki,...li->...kl',stableIdentity(X),stableIdentity(Y))**2-1
# 	#loss = tr.acos(prod.clamp(min=-1.,max=1.))
# 	prod = tr.einsum('...ki,...li->...kl',stableIdentity(X),stableIdentity(Y))**2
# 			#loss = tr.acos(prod.clamp(min=-1.,max=1.))
# 	loss = 2*tr.atan( tr.sqrt(1-  prod.clamp(max=1.)  ) )

# 	return loss



def grad_quaternion_geodesic_dist(X,Y,grad_output):
	eps = 0.
	C = quaternion_a_inv_times_b(X, Y, with_0_comp = False)

	w = tr.norm(C,dim=-1)
	#ww = quaternion_a_inv_times_b(X,Y)
	#w2 = tr.norm(ww,dim=-1)
	mask  = (w>eps)
	ratio = grad_output/w
	weights = tr.zeros_like(grad_output)
	weights[mask] = ratio[mask]
	gradients_x = - tr.einsum('nkl,nkli->nki', weights,C)
	gradients_y = tr.einsum('nkl,nkli->nli', weights,C)
	return gradients_x, gradients_y


# this is correct
def quaternion_prod(a,b):
	shape_a = a.shape
	c = tr.zeros_like(a)
	c[:,:,1:] =  tr.cross(a[:,:,1:],b[:,:,1:]) +a[:,:,0].unsqueeze(-1)*b[:,:,1:] + b[:,:,0].unsqueeze(-1)*a[:,:,1:]
	c[:,:,0] = a[:,:,0]*b[:,:,0] -  tr.einsum('ijd,ijd->ij',a[:,:,1:],b[:,:,1:]) 
	return c


# this is also correct
def quaternion_a_inv_times_b(X,Y, with_0_comp = False):
	shape_X = X.shape
	shape_Y = Y.shape
	c = tr.zeros_like(X)
	c = tr.zeros([shape_X[0],shape_X[1],shape_Y[1],shape_X[2]],  dtype = X.dtype, device = X.device )
	inds = [[1,2,3],[2,3,1],[3,1,2]]
	for j in range(3):
		i = inds[j][0]
		i_1 = inds[j][1]
		i_2 = inds[j][2]
		c[:,:,:,i] = tr.einsum('nl,nk->nkl',Y[:,:,i_1],X[:,:,i_2]) -  tr.einsum('nl,nk->nkl',Y[:,:,i_2],X[:,:,i_1]) + tr.einsum('nk,nl->nkl',X[:,:,0],Y[:,:,i]) - tr.einsum('nl,nk->nkl',Y[:,:,0],X[:,:,i])
	if with_0_comp:
		c[:,:,:,0] = tr.einsum('nli,nki->nkl',Y[:,:,:],X[:,:,:])
	return c


# this is also correct
def _quaternion_a_inv_times_b(a,b, with_0_comp = False):
	shape_a = a.shape
	c = tr.zeros_like(a)
	c[:,:,1:] =  tr.cross(b[:,:,1:],a[:,:,1:]) +a[:,:,0].unsqueeze(-1)*b[:,:,1:] - b[:,:,0].unsqueeze(-1)*a[:,:,1:]
	if with_0_comp:
		c[:,:,0] =  tr.einsum('ijd,ijd->ij',a,b)
	return c

# this is also correct
def _norm_im_a_inv_times_b(X,Y):

	# computes the norm of the imaginary part of X^-1*Y

	shape_y = Y.shape
	Matrix = tr.zeros([4,4,4,4],  dtype=Y.dtype, device=Y.device )
	inds = [[1,2,3],[2,3,1],[3,1,2]]
	for i in range(3):
		j = inds[i][0]
		j_1 = inds[i][1]
		j_2 = inds[i][2]
		Matrix[0,0,j,j] += 1
		Matrix[j,j,0,0] += 1
		Matrix[j_2,j_2,j_1,j_1] += 1
		Matrix[j_1,j_1,j_2,j_2] += 1
		
		Matrix[j_1,j_2,j_2,j_1] += -1
		Matrix[j_2,j_1,j_1,j_2] += -1

		Matrix[j_1,j,j_2,0] += 1
		Matrix[j,j_1,0,j_2] += 1

		Matrix[j_1,0,j_2,j] += -1
		Matrix[0,j_1,j,j_2] += -1

		Matrix[j_2,j,j_1,0] += -1
		Matrix[j,j_2,0,j_1] += -1

		Matrix[j_2,0,j_1,j] += 1
		Matrix[0,j_2,j,j_1] += 1

		Matrix[j,0,0,j] += -1
		Matrix[0,j,j,0] += -1



	W_Y = tr.einsum( '...i,...j->...ij', Y, Y )
	W_X = tr.einsum( '...i,...j->...ij', X, X )
	# Finish this 
	tmp = tr.einsum('...ksr,ijsr->...kij', W_X, Matrix) 
	norm_2 = tr.einsum('...lij,...kij->...kl', W_Y, tmp)

	#tmp_2 = tr.einsum('...lsr,ijsr->...lij', W_Y, Matrix)
	#norm_22 = tr.einsum('...kij,...lij->...kl', W_X, tmp_2)

	Weights = tr.sqrt(norm_2.clamp(0))
	return Weights 
















