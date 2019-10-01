
import  torch as tr
from torch.autograd import Variable
import torch.nn as nn
from torch import autograd

from kernel.gaussian import *
from Utils import *

import os
import torch.nn.functional as F



class Identity(nn.Module):
	def __init__(self):
		super(Identity,self).__init__()
	def forward(self,x):
		return x

class quadexp(nn.Module):
	def __init__(self, sigma = 2.):
		super(quadexp,self).__init__()
		self.sigma = sigma
	def forward(self,x):
		return tr.exp(-x**2/(self.sigma**2))

class quadratic(nn.Module):
	def __init__(self, sigma = 1.):
		super(quadratic,self).__init__()
		self.sigma = sigma
	def forward(self,x):
		return x



class NoisyLinear(nn.Linear):
	def __init__(self, in_features, out_features, noise_level=1., noise_decay = 0.1, bias=False):
		super(NoisyLinear, self).__init__(in_features, out_features, bias=bias)
		
		#self.noise_level = Variable(tr.Tensor([noise_level]), requires_grad = False)
		self.noise_level = noise_level
		self.register_buffer("epsilon_weight", tr.zeros(out_features, in_features))
		if bias:
			self.register_buffer("epsilon_bias", tr.zeros(out_features))
		#self.reset_parameters()
		self.noisy_mode = False
		self.noise_decay = noise_decay



	def update_noise_level(self):
		self.noise_level = self.noise_decay * self.noise_level
	def set_noisy_mode(self,is_noisy):
		self.noisy_mode = is_noisy

	def forward(self, input):
		if self.noisy_mode:
			tr.randn(self.epsilon_weight.size(), out=self.epsilon_weight)
			bias = self.bias
			if bias is not None:
				tr.randn(self.epsilon_bias.size(), out=self.epsilon_bias)
				bias = bias + self.noise_level * Variable(self.epsilon_bias, requires_grad  = False)
			self.noisy_mode = False
			return F.linear(input, self.weight + self.noise_level * Variable(self.epsilon_weight, requires_grad=False), bias)
		else:
			return F.linear(input, self.weight , self.bias)
	def add_noise(self):
		tr.randn(self.epsilon_weight.size(), out=self.epsilon_weight)
		self.weight.data +=  self.noise_level * Variable(self.epsilon_weight, requires_grad=False)
		bias = self.bias
		if bias is not None:
			tr.randn(self.epsilon_bias.size(), out=self.epsilon_bias)
			self.bias.data += self.noise_level * Variable(self.epsilon_bias, requires_grad  = False)
			
		#return F.linear(input, self.weight + self.noise_level * Variable(self.epsilon_weight, requires_grad=False), bias)
		#else:
		#return F.linear(input, self.weight , self.bias)

class OneHiddenLayer(nn.Module):
	def __init__(self,d_int, H, d_out,non_linearity = Identity(),bias=False):
		super(OneHiddenLayer,self).__init__()

		"""
		In the constructor we instantiate two nn.Linear modules and assign them as
		member variables.
		"""
		self.linear1 = tr.nn.Linear(d_int, H,bias=bias)
		self.linear2 = tr.nn.Linear(H, d_out,bias=bias)
		#self.softmax = tr.nn.Softmax()
		self.non_linearity = non_linearity
		self.d_int = d_int
		self.d_out = d_out

	def weights_init(self,center, std):
		self.linear1.weights_init(center,std)
		self.linear2.weights_init(center,std)


	def forward(self, x):
		"""
		In the forward function we accept a Tensor of input data and we must return
		a Tensor of output data. We can use Modules defined in the constructor as
		well as arbitrary operators on Tensors.
		"""
		h1_relu = self.linear1(x).clamp(min=0)
		h2_relu = self.linear2(h1_relu)
		h2_relu = self.non_linearity(h2_relu)

		return h2_relu


class TestOneHiddenLayer(nn.Module):
	def __init__(self,network):
		super(TestOneHiddenLauer,self).__init__()
		self.network = network
		self.noisy_mode = False
	def set_noisy_mode(self,is_noisy):
		self.noisy_mode = is_noisy

	def update_noise_level(self):
		self.noisy_mode = False
	def forward(self,x):
		return self.network(x).unsqueeze(-1)


class NoisyOneHiddenLayer(nn.Module):
	def __init__(self,d_int, H, d_out, n_particles,non_linearity = Identity(),noise_level=1., noise_decay = 0.1,bias=False):
		super(NoisyOneHiddenLayer,self).__init__()

		"""
		In the constructor we instantiate two nn.Linear modules and assign them as
		member variables.
		"""
		self.linear1 = NoisyLinear(d_int, H*n_particles,noise_level = noise_level,noise_decay=noise_decay,bias=bias)
		self.linear2 = NoisyLinear(H*n_particles, n_particles*d_out,noise_level = noise_level,noise_decay=noise_decay,bias= bias)

		#self.linear1 = NoisyLinear(d_int, H,noise_level = noise_level,noise_decay=noise_decay)
		#self.linear2 = NoisyLinear(H, d_out,noise_level = noise_level,noise_decay=noise_decay)


		self.non_linearity = non_linearity
		self.n_particles = n_particles
		self.d_out = d_out

	def set_noisy_mode(self,is_noisy):
		self.linear1.set_noisy_mode(is_noisy)
		self.linear2.set_noisy_mode(is_noisy)

	def update_noise_level(self):
		self.linear1.update_noise_level()
		self.linear2.update_noise_level()

	def weights_init(self,center, std):
		self.linear1.weights_init(center,std)
		self.linear2.weights_init(center,std)

	def forward(self, x):
		"""
		In the forward function we accept a Tensor of input data and we must return
		a Tensor of output data. We can use Modules defined in the constructor as
		well as arbitrary operators on Tensors.
		"""
		h1_relu = self.linear1(x).clamp(min=0)
		h2_relu = self.linear2(h1_relu)
		h2_relu = h2_relu.view(-1,self.d_out, self.n_particles)
		h2_relu = self.non_linearity(h2_relu)
		#h2_relu = h2_relu**2

		return h2_relu
	def add_noise(self):
		self.linear1.add_noise()
		self.linear2.add_noise()

class SphericalTeacher(tr.utils.data.Dataset):

	def __init__(self,network, N_samples, dtype, device):
		#super(Teacher,self).__init__()		
		D = network.d_int
		cpu_device = device
		self.device = device
		self.source = tr.distributions.multivariate_normal.MultivariateNormal(tr.zeros(D ,dtype=dtype,device=cpu_device), tr.eye(D,dtype=dtype,device=cpu_device))
		source_samples = self.source.sample([N_samples])
		inv_norm = 1./tr.norm(source_samples,dim=1)
		self.X = tr.einsum('nd,n->nd',source_samples,inv_norm)
		#inv_norm = 
		#self.X = source_samples/tr.norm(source_samples)
		self.total_size = N_samples
		self.network = network
		#self.network.to(cpu_device)

		with tr.no_grad():
			self.Y = self.network(self.X)
		#self.Y = self.Y.cpu()

	def __len__(self):
		return self.total_size 
	def __getitem__(self,index):
		return self.X[index,:],self.Y[index,:]



class mmd2_random_features(tr.autograd.Function):
	"""
	We can implement our own custom autograd Functions by subclassing
	torch.autograd.Function and implementing the forward and backward passes
	which operate on Tensors.
	"""

	@staticmethod
	def forward(ctx,true_feature,fake_feature,noisy_feature):
		"""
		In the forward pass we receive a Tensor containing the input and return
		a Tensor containing the output. ctx is a context object that can be used
		to stash information for backward computation. You can cache arbitrary
		objects for use in the backward pass using the ctx.save_for_backward method.
		"""
		#noisy_data = fake_data + noise
		#fake_data = fake_data.clone().detach()


		b_size,d, n_particles = noisy_feature.shape
		with  tr.enable_grad():

			mmd2 = tr.mean((true_feature-fake_feature)**2)
			mean_noisy_feature = tr.mean(noisy_feature,dim = -1 )

			mmd2_for_grad = (n_particles/b_size)*(tr.einsum('nd,nd->',fake_feature,mean_noisy_feature) - tr.einsum('nd,nd->',true_feature,mean_noisy_feature))

		ctx.save_for_backward(mmd2_for_grad,noisy_feature)

		return 0.5*mmd2

	@staticmethod
	def backward(ctx, grad_output):
		"""
		In the backward pass we receive a Tensor containing the gradient of the loss
		with respect to the output, and we need to compute the gradient of the loss
		with respect to the input.
		"""
		mmd2_for_grad, noisy_feature = ctx.saved_tensors
		with  tr.enable_grad():
			gradients = autograd.grad(outputs=mmd2_for_grad, inputs=noisy_feature,
						grad_outputs=grad_output,
						create_graph=True, only_inputs=True)[0] 
				
		return None, None, gradients


class mmd2_random_features_no_smoothing(tr.autograd.Function):
	"""
	We can implement our own custom autograd Functions by subclassing
	torch.autograd.Function and implementing the forward and backward passes
	which operate on Tensors.
	"""

	@staticmethod
	def forward(ctx,true_feature,fake_feature):
		"""
		In the forward pass we receive a Tensor containing the input and return
		a Tensor containing the output. ctx is a context object that can be used
		to stash information for backward computation. You can cache arbitrary
		objects for use in the backward pass using the ctx.save_for_backward method.
		"""
		#noisy_data = fake_data + noise
		#fake_data = fake_data.clone().detach()

		b_size,d, n_particles = fake_feature.shape

		with  tr.enable_grad():

			mmd2 = (0.5*n_particles/b_size)*tr.sum((true_feature-tr.mean(fake_feature,dim=-1))**2)

		ctx.save_for_backward(mmd2,fake_feature)

		return (1./n_particles)*mmd2

	@staticmethod
	def backward(ctx, grad_output):
		"""
		In the backward pass we receive a Tensor containing the gradient of the loss
		with respect to the output, and we need to compute the gradient of the loss
		with respect to the input.
		"""
		mmd2, fake_feature = ctx.saved_tensors
		with  tr.enable_grad():
			gradients = autograd.grad(outputs=mmd2, inputs=fake_feature,
						grad_outputs=grad_output,
						create_graph=True, only_inputs=True)[0] 
				
		return None, gradients


class sobolev(tr.autograd.Function):
	"""
	We can implement our own custom autograd Functions by subclassing
	torch.autograd.Function and implementing the forward and backward passes
	which operate on Tensors.
	"""

	@staticmethod
	def forward(ctx,true_feature,fake_feature,matrix):
		"""
		In the forward pass we receive a Tensor containing the input and return
		a Tensor containing the output. ctx is a context object that can be used
		to stash information for backward computation. You can cache arbitrary
		objects for use in the backward pass using the ctx.save_for_backward method.
		"""
		#noisy_data = fake_data + noise
		#fake_data = fake_data.clone().detach()

		b_size,_, n_particles = fake_feature.shape

		m = tr.mean(fake_feature,dim=-1) -  true_feature

		alpha = tr.gesv(m,matrix)[0].clone().detach()

	


		with  tr.enable_grad():

			mmd2 = (0.5*n_particles/b_size)*tr.sum((true_feature-tr.mean(fake_feature,dim=-1))**2)
			mmd2_for_grad = (1./b_size)*tr.einsum('id,idm->',alpha,fake_feature)
		
		ctx.save_for_backward(mmd2_for_grad,fake_feature)

		return (1./n_particles)*mmd2

	@staticmethod
	def backward(ctx, grad_output):
		"""
		In the backward pass we receive a Tensor containing the gradient of the loss
		with respect to the output, and we need to compute the gradient of the loss
		with respect to the input.
		"""
		mmd2, fake_feature = ctx.saved_tensors
		with  tr.enable_grad():
			gradients = autograd.grad(outputs=mmd2, inputs=fake_feature,
						grad_outputs=grad_output,
						create_graph=True, only_inputs=True)[0] 
				
		return None, gradients,None


class SimpleLoss(nn.Module):
	def __init__(self,student):
		super(SimpleLoss, self).__init__()
		self.student = student
	def forward(self,x,y):
		fake_feature = tr.mean(self.student(x),dim=-1)
		mmd2 = tr.mean((y-fake_feature)**2)

		return 0.5*mmd2


class Loss(nn.Module):
	def __init__(self,student):
		super(Loss, self).__init__()
		self.student = student
		self.mmd2 = mmd2_random_features.apply
	def forward(self,x,y):
		out = tr.mean(self.student(x),dim = -1).clone().detach()
		self.student.set_noisy_mode(True)
		noisy_out = self.student(x)
		
		loss = self.mmd2(y,out,noisy_out)
		return loss

class LossDiffusion(nn.Module):
	def __init__(self,student):
		super(LossDiffusion, self).__init__()
		self.student = student
		self.mmd2 = mmd2_random_features_no_smoothing.apply
	def forward(self,x,y):
		#self.student.set_noisy_mode(True)
		#out = tr.mean(self.student(x),dim = -1).clone().detach()
		#self.student.set_noisy_mode(True)
		self.student.add_noise()
		noisy_out = self.student(x)
		
		loss = self.mmd2(y,noisy_out)
		return loss


class LossSobolev(nn.Module):
	def __init__(self,student):
		super(LossSobolev, self).__init__()
		self.student = student
		self.sobolev = sobolev.apply
		self.lmbda = 1e-6
	def forward(self,x,y):
		#self.student.set_noisy_mode(True)
		#out = tr.mean(self.student(x),dim = -1).clone().detach()
		#self.student.set_noisy_mode(True)
		#self.student.add_noise()
		self.student.zero_grad()
		out = self.student(x)
		b_size,_,num_particles = out.shape
		grad_out = compute_grad(self.student,x)
		#grad_out = tr.zeros([b_size,1],dtype=x.dtype,device=x.device)
		#grad_out = tr.autograd.grad(outputs=out, inputs=self.student.parameters(),create_graph=False)
		matrix = (1./(num_particles*b_size))*tr.einsum('im,jm->ij',grad_out,grad_out)+self.lmbda*tr.eye(b_size, dtype= x.dtype, device=x.device)
		matrix = matrix.clone().detach()
		loss = self.sobolev(y,out,matrix)
		return loss


def compute_grad(net,x):
	J = []
	F = net(x)
	F = tr.einsum('idm->i',F)
	b_size = F.shape[0]
	for i in range(b_size):
		if i==b_size-1:
			grads =  tr.autograd.grad(F[i], net.parameters(),retain_graph=False)
		else:
			grads =  tr.autograd.grad(F[i], net.parameters(),retain_graph=True)
		grads = [x.view(-1) for x in grads]
		grads = tr.cat(grads)
		J.append(grads)

	return tr.stack(J,dim=0)





