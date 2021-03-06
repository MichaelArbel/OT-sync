import  torch as tr
from torch.autograd import Variable
import torch.nn as nn
from torch import autograd
from utils import *
import os
import numpy as np

import utils


import torch
from torch.optim import Optimizer
import torch.optim as optim
from copy import deepcopy

class quaternion_SGD(Optimizer):

	def __init__(self, params, lr=0.1, weights_factor=0.001 , momentum=0, dampening=0,
				 weight_decay=0, nesterov=False):
		if lr < 0.0:
			raise ValueError("Invalid learning rate: {}".format(lr))
		if momentum < 0.0:
			raise ValueError("Invalid momentum value: {}".format(momentum))
		if weight_decay < 0.0:
			raise ValueError("Invalid weight_decay value: {}".format(weight_decay))

		defaults = dict(lr=lr, momentum=momentum, dampening=dampening,
						weight_decay=weight_decay, nesterov=nesterov)
		if nesterov and (momentum <= 0 or dampening != 0):
			raise ValueError("Nesterov momentum requires a momentum and zero dampening")
		super(quaternion_SGD, self).__init__(params, defaults)

		self.params = None
		self.weights_factor = weights_factor

	def __setstate__(self, state):
		super(quaternion_SGD, self).__setstate__(state)
		for group in self.param_groups:
			group.setdefault('nesterov', False)

	def step(self, loss=None, closure=None):
		"""Performs a single optimization step.

		Arguments:
			closure (callable, optional): A closure that reevaluates the model
				and returns the loss.
		"""
		if closure is not None:
			loss = closure()

		for group in self.param_groups:
			weight_decay = group['weight_decay']
			momentum = group['momentum']
			dampening = group['dampening']
			nesterov = group['nesterov']

			for i, param in enumerate(group['params']):
				if param.grad is None:
					continue
				d_p = param.grad.data
				if weight_decay != 0:
					d_p.add_(weight_decay, param.data)
				if momentum != 0:
					param_state = self.state[p]
					if 'momentum_buffer' not in param_state:
						buf = param_state['momentum_buffer'] = torch.clone(d_p).detach()
					else:
						buf = param_state['momentum_buffer']
						buf.mul_(momentum).add_(1 - dampening, d_p)
					if nesterov:
						d_p = d_param.add(momentum, buf)
					else:
						d_p = buf
				if i==0:
					#v = - group['lr']* utils.quaternion_proj(param.data,d_p)
					effective_lr = compute_lr(group['lr'],d_p,loss)
					v = - effective_lr* d_p
					param.data = utils.quaternion_exp_map(param.data,v, north_hemisphere=False)
					param.data[0,:,0] = 1.
					param.data[0,:,1:] = 0.

				elif i==1:
					effective_lr = compute_lr(group['lr'],d_p,loss)
					v = - self.weights_factor*effective_lr* utils.sphere_proj(param.data,d_p)
					param.data = utils.sphere_exp_map(param.data,v, north_hemisphere=False)
				else:
					effective_lr = compute_lr(group['lr'],d_p,loss)
					param.data -=  self.weights_factor*effective_lr*d_p
					#print(param.daat)
					#w_g = - group['lr']* (d_p - tr.sum(d_p,dim=-1).unsqueeze(-1))
					#param.data = (param.data + w_g).clamp(0.)
					#param.data = param.data/tr.sum(param.data, dim=-1).unsqueeze(-1)
		return loss
	def keep_weights(self):

		self.params = deepcopy(self.param_groups)
	def reset_weights(self):
		for k ,group in enumerate(self.param_groups):
			for i, param in enumerate(group['params']):
				param.data = deepcopy(self.params[k]['params'][i].data)
	def decrease_lr(self):
		for k ,group in enumerate(self.param_groups):
			group['lr']*=0.1	
	def reset_lr(self,lr):
		for k ,group in enumerate(self.param_groups):
			group['lr'] =lr		


class quaternion_SGD_unconstrained(Optimizer):

	def __init__(self, params, lr=0.1, weights_factor=0.001 , momentum=0, dampening=0,
				 weight_penalty=0, nesterov=False):
		if lr < 0.0:
			raise ValueError("Invalid learning rate: {}".format(lr))
		if momentum < 0.0:
			raise ValueError("Invalid momentum value: {}".format(momentum))
		if weight_penalty < 0.0:
			raise ValueError("Invalid weight_decay value: {}".format(weight_penalty))

		defaults = dict(lr=lr, momentum=momentum, dampening=dampening,
						weight_decay=0., nesterov=nesterov)
		if nesterov and (momentum <= 0 or dampening != 0):
			raise ValueError("Nesterov momentum requires a momentum and zero dampening")
		super(quaternion_SGD_unconstrained, self).__init__(params, defaults)

		self.params = None
		self.weights_factor = weights_factor
		self.weight_penalty = weight_penalty
	def __setstate__(self, state):
		super(quaternion_SGD_unconstrained, self).__setstate__(state)
		for group in self.param_groups:
			group.setdefault('nesterov', False)

	def step(self, loss=None, closure=None):
		"""Performs a single optimization step.

		Arguments:
			closure (callable, optional): A closure that reevaluates the model
				and returns the loss.
		"""
		if closure is not None:
			loss = closure()

		for group in self.param_groups:
			weight_decay = group['weight_decay']
			momentum = group['momentum']
			dampening = group['dampening']
			nesterov = group['nesterov']

			for i, param in enumerate(group['params']):
				if param.grad is None:
					continue
				d_p = param.grad.data
				if weight_decay != 0:
					d_p.add_(weight_decay, param.data)
				if momentum != 0:
					param_state = self.state[p]
					if 'momentum_buffer' not in param_state:
						buf = param_state['momentum_buffer'] = torch.clone(d_p).detach()
					else:
						buf = param_state['momentum_buffer']
						buf.mul_(momentum).add_(1 - dampening, d_p)
					if nesterov:
						d_p = d_param.add(momentum, buf)
					else:
						d_p = buf
				if i==0:
					#v = - group['lr']* utils.quaternion_proj(param.data,d_p)
					effective_lr = compute_lr(group['lr'],d_p,loss)
					v = - effective_lr* d_p
					param.data = utils.quaternion_exp_map(param.data,v, north_hemisphere=False)
					param.data[0,:,0] = 1.
					param.data[0,:,1:] = 0.

				elif i==1:
					effective_lr = compute_lr(group['lr'],d_p,loss)
					#v = - self.weights_factor*effective_lr* utils.sphere_proj(param.data,d_p)
					param.data -=  self.weights_factor*effective_lr * d_p + self.penalty(param.data)
				else:
					effective_lr = compute_lr(group['lr'],d_p,loss)
					param.data -=  self.weights_factor*effective_lr*d_p
					#print(param.daat)
					#w_g = - group['lr']* (d_p - tr.sum(d_p,dim=-1).unsqueeze(-1))
					#param.data = (param.data + w_g).clamp(0.)
					#param.data = param.data/tr.sum(param.data, dim=-1).unsqueeze(-1)
		return loss
	def keep_weights(self):

		self.params = deepcopy(self.param_groups)
	def reset_weights(self):
		for k ,group in enumerate(self.param_groups):
			for i, param in enumerate(group['params']):
				param.data = deepcopy(self.params[k]['params'][i].data)
	def decrease_lr(self):
		for k ,group in enumerate(self.param_groups):
			group['lr']*=0.1	
	def reset_lr(self,lr):
		for k ,group in enumerate(self.param_groups):
			group['lr'] =lr	
	def penalty(self,params):
		return self.weight_penalty*params







def compute_lr(lr, v, loss):
	if loss is None:
		return lr
	else:
		tmp  =  loss/tr.norm(v)**2
	#return lr
		return min(lr,tmp)

