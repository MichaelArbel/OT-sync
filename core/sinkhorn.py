import torch
import torch.nn as nn
import torchvision.datasets as dsets
from torchvision import datasets, transforms
from torch.autograd import Variable
from torch.nn import functional as F
import numpy as np
import pandas as pd
from utils import quaternion_geodesic_distance, squared_quaternion_geodesic_distance
import utils
import math

# Adapted from ../OptimalTransportSync
# Adapted from https://github.com/gpeyre/SinkhornAutoDiff

from geomloss import SamplesLoss
import sinkhorn_divergence as sd

from functools import partial
import torch as tr
from torch import autograd

def squared_dist(x,y):
	tmp = (X.unsqueeze(-2) - Y.unsqueeze(-3))**2
	dist =  tr.sum(tmp,dim=-1)
	return dist

def get_loss(kernel,eps):
	if kernel.kernel_type=='quaternion':
		return SamplesLoss("sinkhorn", blur=eps, diameter=3.15, cost = utils.quaternion_geodesic_distance, backend= 'tensorized')
	elif kernel.kernel_type=='squared_euclidean':
		return SamplesLoss("sinkhorn", p=2, blur=eps, diameter=4., backend= 'tensorized')
	elif kernel.kernel_type=='power_quaternion':
		dist = partial(utils.power_quaternion_geodesic_distance,kernel.power)
		#dist = partial(utils.sum_power_quaternion_geodesic_distance,kernel.power)
		#return sinkhorn_wasserstein_fisher_rao,dist
		return SamplesLoss("sinkhorn", blur=eps, diameter=10.,cost = dist, backend= 'tensorized')
	elif kernel.kernel_type=='sum_power_quaternion':
		dist = partial(utils.sum_power_quaternion_geodesic_distance,kernel.power)
		return SamplesLoss("sinkhorn", blur=eps, diameter=10.,cost = dist, backend= 'tensorized')

def get_loss_w_fisher(kernel,eps):
	if kernel.kernel_type=='quaternion':
		dist = utils.quaternion_geodesic_distance
	elif kernel.kernel_type=='squared_euclidean':
		dist = squared_dist
	elif kernel.kernel_type=='power_quaternion':
		dist = partial(utils.power_quaternion_geodesic_distance,kernel.power)
	return sinkhorn_wasserstein_fisher_rao,dist





	#elif kernel.kernel_type=='sinkhorn_gaussian':
	#    return SamplesLoss("gaussian", blur=1., diameter=4., backend= 'tensorized')
	#elif kernel.kernel_type=='min_squared_euclidean':
	#   return SamplesLoss("sinkhorn", blur=eps, diameter=10.,cost = utils.min_squared_eucliean_distance, backend= 'tensorized')
#loss = SamplesLoss("sinkhorn", blur=.05, diameter=3.15, cost = quaternion_geodesic_distance_geomloss, backend= 'tensorized')


class Sinkhorn(nn.Module):
	r"""
	Given two empirical measures each with :math:`P_1` locations
	:math:`x\in\mathbb{R}^{D_1}` and :math:`P_2` locations :math:`y\in\mathbb{R}^{D_2}`,
	outputs an approximation of the regularized OT cost for point clouds.
	Args:
		eps (float): regularization coefficient
		max_iter (int): maximum number of Sinkhorn iterations
		reduction (string, optional): Specifies the reduction to apply to the output:
			'none' | 'mean' | 'sum'. 'none': no reduction will be applied,
			'mean': the sum of the output will be divided by the number of
			elements in the output, 'sum': the output will be summed. Default: 'none'
	Shape:
		- Input: :math:`(N, P_1, D_1)`, :math:`(N, P_2, D_2)`
		- Output: :math:`(N)` or :math:`()`, depending on `reduction`
	"""
	def __init__(self, kernel, particles,rm_map, eps=0.05):
		super(Sinkhorn, self).__init__()
		self.eps = eps
		self.kernel = kernel
		self.particles = particles
		self.rm_map = rm_map
		self.loss = get_loss(kernel,eps)
	def forward(self, true_data):
		# The Sinkhorn algorithm takes as input three variables :
		y = self.rm_map(self.particles.data)
		return  torch.sum(self.loss(true_data,y))


class Sinkhorn_weighted(nn.Module):
	r"""
	Given two empirical measures each with :math:`P_1` locations
	:math:`x\in\mathbb{R}^{D_1}` and :math:`P_2` locations :math:`y\in\mathbb{R}^{D_2}`,
	outputs an approximation of the regularized OT cost for point clouds.
	Args:
		eps (float): regularization coefficient
		max_iter (int): maximum number of Sinkhorn iterations
		reduction (string, optional): Specifies the reduction to apply to the output:
			'none' | 'mean' | 'sum'. 'none': no reduction will be applied,
			'mean': the sum of the output will be divided by the number of
			elements in the output, 'sum': the output will be summed. Default: 'none'
	Shape:
		- Input: :math:`(N, P_1, D_1)`, :math:`(N, P_2, D_2)`
		- Output: :math:`(N)` or :math:`()`, depending on `reduction`
	"""
	def __init__(self, kernel, particles,rm_map, eps=0.05):
		super(Sinkhorn_weighted, self).__init__()
		self.eps = eps
		self.kernel = kernel
		self.particles = particles
		self.rm_map = rm_map
		self.loss,self.cost = get_loss_w_fisher(kernel,eps)
		self.diameter = 10.
		self.scaling = 0.5
		self.a_x = 0.
		self.b_y = 0.
		self.a_y = 0.
		self.b_x = 0.
	def forward(self, true_data,true_weights,edges):
		# The Sinkhorn algorithm takes as input three variables :
		x, weights_x = self.rm_map(self.particles.data,self.particles.weights(),edges)
		y = true_data
		weights_y = true_weights
		cost = self.cost

		diameter = 10.
		scaling = 0.5
		blur = 0.001
		p = 1.1
		#with  tr.enable_grad():

		#out = torch.sum(self.loss(true_weights,true_data,weights,y))
		diameter, eps, eps_s, rho = sd.scaling_parameters( x, y, p, blur, None, diameter, scaling )
		C_xx, C_yy = ( cost( x, x.detach()), cost( y, y.detach()) )   # (B,N,N), (B,M,M)
		C_xy, C_yx = ( cost( x, y.detach()), cost( y, x.detach()) )  # (B,N,M), (B,M,N)

		a_x, b_y, a_y, b_x = sd.sinkhorn_loop( softmin_tensorized, 
									sd.log_weights(weights_x), sd.log_weights(weights_y), 
									C_xx, C_yy, C_xy, C_yx, eps_s, rho, debias=True, a_x_0 = self.a_x,a_y_0=self.a_y, b_x_0=self.b_x, b_y_0=self.b_y)
		#self.update_dual(a_x,b_y,a_y,b_x)
		out = sd.sinkhorn_cost(eps, rho, weights_x, weights_y, a_x, b_y, a_y, b_x, batch=True, debias=True, potentials=False)
		out = tr.sum(out)
		#out_x = tr.sum(tr.einsum('nl,nl->n',weights_x,(b_x-a_x)))
		out_x = tr.sum(b_x-a_x)


		return sinkhorn_wasserstein_fisher_rao(out,out_x,x,weights_x)
	def update_dual(self,a_x,b_y,a_y,b_x):
		self.a_x = a_x
		self.b_y = b_y
		self.a_y = a_y
		self.b_x = b_x

		#return  torch.sum(self.loss(self.cost,true_weights,true_data,weights,y,self.eps, self.diameter,self.scaling))






class Sinkhorn_weighted(nn.Module):
	r"""
	Given two empirical measures each with :math:`P_1` locations
	:math:`x\in\mathbb{R}^{D_1}` and :math:`P_2` locations :math:`y\in\mathbb{R}^{D_2}`,
	outputs an approximation of the regularized OT cost for point clouds.
	Args:
		eps (float): regularization coefficient
		max_iter (int): maximum number of Sinkhorn iterations
		reduction (string, optional): Specifies the reduction to apply to the output:
			'none' | 'mean' | 'sum'. 'none': no reduction will be applied,
			'mean': the sum of the output will be divided by the number of
			elements in the output, 'sum': the output will be summed. Default: 'none'
	Shape:
		- Input: :math:`(N, P_1, D_1)`, :math:`(N, P_2, D_2)`
		- Output: :math:`(N)` or :math:`()`, depending on `reduction`
	"""
	def __init__(self, kernel, particles,rm_map, eps=0.05):
		super(Sinkhorn_weighted, self).__init__()
		self.eps = eps
		self.kernel = kernel
		self.particles = particles
		self.rm_map = rm_map
		self.loss = get_loss(kernel,eps)
		self.diameter = 10.
		self.scaling = 0.5
		self.x = None
		self.weights_x = None
		self.test = True
	def forward(self, true_data,true_weights,edges):
		# The Sinkhorn algorithm takes as input three variables :
		
		x, weights_x = self.rm_map(self.particles.data,self.particles.weights(),edges)
		
		self.x = x
		self.weights_x = weights_x
		out = torch.sum(self.loss(true_weights,true_data,weights_x,x))

		# if self.test:
		# 	N = x.shape[0]
		# 	aa  = - 0.01 * autograd.grad(outputs=out, inputs=[self.particles.data],grad_outputs=[tr.tensor(1., dtype=x.dtype, device=x.device)], create_graph=True, only_inputs=True, retain_graph=True)[0]
		# 	#aa = aa.permute([0,3,1,2]).reshape([N,4,-1]).permute([0,2,1])
		# 	#ratios_0 = ratios_0.permute([0,3,1,2]).reshape([N,4,-1]).permute([0,2,1])
		# 	#new_weights_x = utils.sphere_exp_map(self.particles._weights,aa)
		# 	new_x = utils.quaternion_exp_map(self.particles.data,aa)
		# 	#ratios = utils.quaternion_X_times_Y_inv_prod(xi,new_x)
		# 	#new_x = ratios.permute([0,3,1,2]).reshape([N,4,-1]).permute([0,2,1])
		# 	#new_x = new_x.permute([0,3,1,2]).reshape([N,4,-1]).permute([0,2,1])
		# 	new_x, _= self.rm_map(new_x,weights,edges)
		# 	new_out = torch.sum(self.loss(true_weights,true_data,weights_x,new_x))
		# 	err = new_out - out
		# 	err = err.item()
		# 	if err>0:
		# 		print( 'non decreasing loss: '+ str(err))


		return out
		#return sinkhorn_wasserstein_fisher_rao_2(out,x,weights_x)
		  

		#return sinkhorn_wasserstein_fisher_rao(out,out_x,x,weights_x)





class Sinkhorn_wasserstein_fisher_rao(tr.autograd.Function):
	@staticmethod
	def forward(ctx, out,out_x,x,weights_x):
		ctx.save_for_backward(out,out_x,x,weights_x)

		return out
	@staticmethod
	def backward(ctx, grad_output):

		out,out_x,x,weights_x = ctx.saved_tensors
		with  tr.enable_grad():
		#return aa
			gradients_x  = autograd.grad(outputs=out_x, inputs=[x],grad_outputs=grad_output, create_graph=True, only_inputs=True)[0]
		#gradients_x = gradients_x/weights.unsqueeze(-1)
		#gradients_x

			if weights_x.requires_grad:
				gradients_y = autograd.grad(outputs=out, inputs=[weights_x],
						grad_outputs=grad_output,
						create_graph=True, only_inputs=True)[0]
			else:
				gradients_y = None
		#return aa
		return None, None,gradients_x,gradients_y



class Sinkhorn_wasserstein_fisher_rao_2(tr.autograd.Function):
	@staticmethod
	def forward(ctx, out,x,weights_x):
		ctx.save_for_backward(out,x,weights_x)

		return out
	@staticmethod
	def backward(ctx, grad_output):

		out,x,weights_x = ctx.saved_tensors
		with  tr.enable_grad():
		#return aa
			gradients_x  = autograd.grad(outputs=out, inputs=[x],grad_outputs=grad_output, create_graph=True, only_inputs=True)[0]
		#gradients_x = gradients_x/weights.unsqueeze(-1)
		#gradients_x
			inv_w = 1./weights_x
			inv_w[weights_x<=0.] = 1.
			gradients_x = gradients_x*inv_w.unsqueeze(-1)
			if weights_x.requires_grad:
				gradients_y = autograd.grad(outputs=out, inputs=[weights_x],
						grad_outputs=grad_output,
						create_graph=True, only_inputs=True)[0]
			else:
				gradients_y = None
		#return aa
		return None,gradients_x,gradients_y

sinkhorn_wasserstein_fisher_rao_2 = Sinkhorn_wasserstein_fisher_rao_2.apply


sinkhorn_wasserstein_fisher_rao = Sinkhorn_wasserstein_fisher_rao.apply

# class Sinkhorn_weighted(nn.Module):
#     r"""
#     Given two empirical measures each with :math:`P_1` locations
#     :math:`x\in\mathbb{R}^{D_1}` and :math:`P_2` locations :math:`y\in\mathbb{R}^{D_2}`,
#     outputs an approximation of the regularized OT cost for point clouds.
#     Args:
#         eps (float): regularization coefficient
#         max_iter (int): maximum number of Sinkhorn iterations
#         reduction (string, optional): Specifies the reduction to apply to the output:
#             'none' | 'mean' | 'sum'. 'none': no reduction will be applied,
#             'mean': the sum of the output will be divided by the number of
#             elements in the output, 'sum': the output will be summed. Default: 'none'
#     Shape:
#         - Input: :math:`(N, P_1, D_1)`, :math:`(N, P_2, D_2)`
#         - Output: :math:`(N)` or :math:`()`, depending on `reduction`
#     """
#     def __init__(self, kernel, particles,rm_map, eps=0.05):
#         super(Sinkhorn_weighted, self).__init__()
#         self.eps = eps
#         self.kernel = kernel
#         self.particles = particles
#         self.rm_map = rm_map
#         self.loss = get_loss(kernel,eps)
#     def forward(self, true_data,true_weights):
#         # The Sinkhorn algorithm takes as input three variables :
#         y, weights = self.rm_map(self.particles.data,self.particles.weights())
#         return  torch.sum(utils.power_quaternion_geodesic_distance(self.particles.data,true_data))



# class SinkhornLoss(nn.Module):
#     r"""
#     Given two empirical measures each with :math:`P_1` locations
#     :math:`x\in\mathbb{R}^{D_1}` and :math:`P_2` locations :math:`y\in\mathbb{R}^{D_2}`,
#     outputs an approximation of the regularized OT cost for point clouds.
#     Args:
#         eps (float): regularization coefficient
#         max_iter (int): maximum number of Sinkhorn iterations
#         reduction (string, optional): Specifies the reduction to apply to the output:
#             'none' | 'mean' | 'sum'. 'none': no reduction will be applied,
#             'mean': the sum of the output will be divided by the number of
#             elements in the output, 'sum': the output will be summed. Default: 'none'
#     Shape:
#         - Input: :math:`(N, P_1, D_1)`, :math:`(N, P_2, D_2)`
#         - Output: :math:`(N)` or :math:`()`, depending on `reduction`
#     """
#     def __init__(self, kernel_type, particles,rm_map, eps=0.05):
#         super(SinkhornLoss, self).__init__()
#         self.eps = eps
#         self.kernel_type = kernel_type
#         self.particles = particles
#         self.rm_map = rm_map
#         self.loss = get_loss(kernel_type,eps)
#     def forward(self, true_data):
#         # The Sinkhorn algorithm takes as input three variables :
#         y = self.particles.data
#         #print(true_data)
#         return  torch.sum(self.loss(true_data,y))


def softmin_tensorized(eps, C, f):
	B = C.shape[0]
	return - eps * ( f.view(B,1,-1) - C/eps ).logsumexp(2).view(B,-1)

class SinkhornEval(nn.Module):
	r"""
	Given two empirical measures each with :math:`P_1` locations
	:math:`x\in\mathbb{R}^{D_1}` and :math:`P_2` locations :math:`y\in\mathbb{R}^{D_2}`,
	outputs an approximation of the regularized OT cost for point clouds.
	Args:
		eps (float): regularization coefficient
		max_iter (int): maximum number of Sinkhorn iterations
		reduction (string, optional): Specifies the reduction to apply to the output:
			'none' | 'mean' | 'sum'. 'none': no reduction will be applied,
			'mean': the sum of the output will be divided by the number of
			elements in the output, 'sum': the output will be summed. Default: 'none'
	Shape:
		- Input: :math:`(N, P_1, D_1)`, :math:`(N, P_2, D_2)`
		- Output: :math:`(N)` or :math:`()`, depending on `reduction`
	"""
	def __init__(self,  particles, rm_map, eps, max_iter, particles_type):
		super(SinkhornEval, self).__init__()
		self.eps = eps
		self.particles_type = particles_type
		self.particles = particles
		self.RM_map = rm_map
		self.loss = SamplesLoss("sinkhorn", blur=eps, diameter=3.15, cost = utils.quaternion_geodesic_distance, backend= 'tensorized')

	def forward(self, y,w_y,edges):
		# The Sinkhorn algorithm takes as input three variables :
		x, w_x =self.RM_map(self.particles.data, self.particles.weights(),edges )
		if w_x is None or w_y is None:

			return  torch.sum(self.loss(x,y))
		else:
			return  torch.sum(self.loss(w_x,x,w_y,y))
class SinkhornEvalAbs(nn.Module):
	r"""
	Given two empirical measures each with :math:`P_1` locations
	:math:`x\in\mathbb{R}^{D_1}` and :math:`P_2` locations :math:`y\in\mathbb{R}^{D_2}`,
	outputs an approximation of the regularized OT cost for point clouds.
	Args:
		eps (float): regularization coefficient
		max_iter (int): maximum number of Sinkhorn iterations
		reduction (string, optional): Specifies the reduction to apply to the output:
			'none' | 'mean' | 'sum'. 'none': no reduction will be applied,
			'mean': the sum of the output will be divided by the number of
			elements in the output, 'sum': the output will be summed. Default: 'none'
	Shape:
		- Input: :math:`(N, P_1, D_1)`, :math:`(N, P_2, D_2)`
		- Output: :math:`(N)` or :math:`()`, depending on `reduction`
	"""
	def __init__(self,  particles, eps, max_iter, particles_type,eval_idx):
		super(SinkhornEvalAbs, self).__init__()
		self.eps = eps
		self.particles_type = particles_type
		self.particles = particles
		self.eval_idx = eval_idx
		#self.RM_map = rm_map
		self.loss = SamplesLoss("sinkhorn", blur=eps, diameter=3.15, cost = utils.quaternion_geodesic_distance, backend= 'tensorized')
		
	def forward(self, y,w_y):
		# The Sinkhorn algorithm takes as input three variables :
		x, w_x = self.particles.data, self.particles.weights()
		if self.eval_idx is None:
			if w_x is None or w_y is None:
				return  torch.mean(self.loss(x,y))
			else:

				return  torch.mean(self.loss(w_x,x,w_y,y))
		else:
			if w_x is None or w_y is None:
				return  torch.mean(self.loss(x[self.eval_idx,:,:],y[self.eval_idx,:,:]))
			else:
				return  torch.mean(self.loss(w_x[self.eval_idx,:],x[self.eval_idx,:,:],w_y[self.eval_idx,:],y[self.eval_idx,:,:]))


class SinkhornEvalKBestAbs(nn.Module):
	r"""
	Given two empirical measures each with :math:`P_1` locations
	:math:`x\in\mathbb{R}^{D_1}` and :math:`P_2` locations :math:`y\in\mathbb{R}^{D_2}`,
	outputs an approximation of the regularized OT cost for point clouds.
	Args:
		eps (float): regularization coefficient
		max_iter (int): maximum number of Sinkhorn iterations
		reduction (string, optional): Specifies the reduction to apply to the output:
			'none' | 'mean' | 'sum'. 'none': no reduction will be applied,
			'mean': the sum of the output will be divided by the number of
			elements in the output, 'sum': the output will be summed. Default: 'none'
	Shape:
		- Input: :math:`(N, P_1, D_1)`, :math:`(N, P_2, D_2)`
		- Output: :math:`(N)` or :math:`()`, depending on `reduction`
	"""

	def __init__(self, particles, eps, max_iter, particles_type, eval_idx):
		super(SinkhornEvalKBestAbs, self).__init__()
		self.eps = eps
		self.particles_type = particles_type
		self.particles = particles
		self.eval_idx = eval_idx
		# self.RM_map = rm_map
		self.loss = SamplesLoss("sinkhorn", blur=eps, diameter=3.15, cost=utils.quaternion_geodesic_distance,
								backend='tensorized')

	def forward(self, y, w_y):
		# The Sinkhorn algorithm takes as input three variables :
		x, w_x = self.particles.data, self.particles.weights()
		if self.eval_idx is None:
			meanDist = 0
			for allDataIndex in range(0, y.size()[0]):
				yi = torch.squeeze(y[allDataIndex, 0, :])
				minD = 9999999999
				for kbestIndex in range(0, x.size()[1]):
					xi = torch.squeeze(x[allDataIndex, kbestIndex, :])
					cost = torch.abs(torch.dot(xi,yi))
					if (cost<-0.99999999):
						cost=torch.tensor([-1.0]).cuda()
					elif(cost>0.9999999):
						cost =torch.tensor([1.0]).cuda()
					d = 2*torch.acos(cost)
					#print(d)
					if (d<minD):
						minD = d
				meanDist = meanDist + minD
			return torch.tensor(meanDist/x.size()[0])
		else:
			if w_x is None or w_y is None:
				return torch.mean(self.loss(x[self.eval_idx, :, :], y[self.eval_idx, :, :]))
			else:
				return torch.mean(self.loss(w_x[self.eval_idx, :], x[self.eval_idx, :, :], w_y[self.eval_idx, :],
											y[self.eval_idx, :, :]))

	# def forward(self, y, w_y):
	# 	# The Sinkhorn algorithm takes as input three variables :
	# 	x, w_x = self.particles.data, self.particles.weights()
	# 	if self.eval_idx is None:
	# 		meanDist = 0
	# 		for allDataIndex in range(0, y.size()[0]):
	# 			yi = torch.squeeze(y[allDataIndex, 0, :])
	# 			minD = 999999
	# 			for kbestIndex in range(0, x.size()[1]):
	# 				xi = torch.squeeze(x[allDataIndex, kbestIndex, :])
	# 				cost = torch.abs(torch.dot(xi,yi))
	# 				if (cost.item()>1.0-0.0000001):
	# 					cost = torch.tensor([1.0])
	# 				elif(cost.item()<=1.0+0.0000001):
	# 					cost = torch.tensor([1.0])
	# 				print(cost)
	# 				d = 2*torch.acos(cost)
	# 				if (d<minD):
	# 					minD = d
	# 			meanDist = meanDist + minD
	# 		return torch.tensor(meanDist/x.size()[0])
	# 	else:
	# 		if w_x is None or w_y is None:
	# 			return torch.mean(self.loss(x[self.eval_idx, :, :], y[self.eval_idx, :, :]))
	# 		else:
	# 			return torch.mean(self.loss(w_x[self.eval_idx, :], x[self.eval_idx, :, :], w_y[self.eval_idx, :],
	# 										y[self.eval_idx, :, :]))
	#
	#