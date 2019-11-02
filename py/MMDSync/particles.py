import  torch as tr
from torch.autograd import Variable
import torch.nn as nn
from torch import autograd
import utils
import numpy as np



def add_noise_quaternion(prior,particles,noise_level):
	N,num_particles,_ = particles.shape 

	noise = prior.sample(N,num_particles)
	
	angle = tr.acos(noise[:,:,0])	
	direction = noise[:,:,1:]/tr.norm(noise[:,:,1:], dim=-1).unsqueeze(-1)
	noise[:,:,0] = tr.cos(noise_level * angle)
	noise[:,:,1:] = tr.einsum( 'np,npd ->npd' ,tr.sin(noise_level * angle) , direction )
	noisy_data = utils.quaternion_prod(particles,noise)
	# pos-hoc normalization
	noisy_data = noisy_data/tr.norm(noisy_data, dim=-1).unsqueeze(-1)
	return noisy_data




class Particles(nn.Module):
	def __init__(self,prior,N, num_particles ,with_weights, product_particles,noise_level , noise_decay,particle_type='euclidian'):
		super(Particles,self).__init__()
		assert prior.type==particle_type
		self.prior = prior
		self.particle_type= particle_type
		self.num_particles = num_particles
		self.N = N
		self.noise_level = noise_level
		self.noise_decay = noise_decay
		self.product_particles = product_particles
		self.with_weights = with_weights
		self.data = nn.Parameter(prior.sample(N,num_particles)) # N x num_particles x d
		if self.product_particles:
			self._weights = (1./np.sqrt(num_particles))*tr.ones([N,num_particles],  dtype=self.data.dtype, device = self.data.device  )
		else:
			self._weights = (1./np.sqrt(num_particles))*tr.ones([num_particles],  dtype=self.data.dtype, device = self.data.device  )
			self._all_weights = tr.ones([N,num_particles],  dtype=self.data.dtype, device = self.data.device )
		if self.with_weights:
			self._weights = nn.Parameter(self._weights)
	def add_noise(self):
		noise = self.prior.sample(self.N,self.num_particles)
		return self.data + self.noise_level*noise 

	def update_noise_level(self):
		self.noise_level *=self.noise_decay
	def weights(self):
		if self.product_particles:
			return self._weights**2
		else:
			return tr.einsum('k,nk->nk', self._weights**2, self._all_weights)
class QuaternionParticles(Particles):
	def __init__(self,prior, N, num_particles, with_weights,product_particles,noise_level, noise_decay):
		# particle is a tensor   of shape N x num_paricles x d
		# where d = 4 is the dimension of a normalized quaternion
		#prior = BinghamGenerator(maxNumModes)
		super(QuaternionParticles,self).__init__(prior,N,num_particles,with_weights,product_particles, noise_level,noise_decay, particle_type='quaternion')

	def add_noise(self):
		noisy_data = add_noise_quaternion(self.prior,self.data,self.noise_level)
		return noisy_data

class RelativeMeasureMap(nn.Module):
	def __init__(self,edges, grad_type='euclidean'):
		super(RelativeMeasureMap,self).__init__()
		self.edges = edges
		self.grad_type = grad_type

	def forward(self,particles):

		ratios = []
		for k in range(self.edges):
			i  = np.int(self.edges[0,k])
			j  = np.int(self.edges[1,k])
			xi = particles[i,:,:]
			xj = particles[j,:,:]
			r  = tr.norm(xi.unsqueeze(1) - xj.unsqueeze(0), dim=-1)
			r  = r.view(-1,1)
			ratios.append(r)
		ratios = tr.stack(ratios,dim=0)
		return ratios


class RelativeMeasureMapWeights(nn.Module):
	def __init__(self,edges,grad_type):
		super(RelativeMeasureMapWeights,self).__init__()
		self.edges = edges
		self.grad_type = grad_type

	def forward(self,particles,weights):
		i = self.edges[0,:]
		j = self.edges[1,:]
		xi = particles[i,:,:]
		xj = particles[j,:,:]
		ratios = xi - xj

		#ratios = tr.stack(ratios,dim=0)
		#RM_weights = weights[i,:]*weights[j,:]
		N = xi.shape[0] 
		RM_weights = weights[0,:].unsqueeze(0).repeat(N,1)

		return ratios,RM_weights


# class QuaternionRelativeMeasureMap(RelativeMeasureMap):
# 	def __init__(self,edges,grad_type='quaternion'):
# 		super(QuaternionRelativeMeasureMap,self).__init__(edges,grad_type)
# 	def forward(self,particles):
# 		ratios = []
# 		i = self.edges[0,:]
# 		j = self.edges[1,:]
# 		xi = particles[i,:,:]
# 		xj = particles[j,:,:]
# 		if self.grad_type=='euclidean':
# 			ratios = utils.forward_quaternion_X_times_Y_inv(xi,xj)
# 		elif self.grad_type=='quaternion':
# 			ratios  = utils.quaternion_X_times_Y_inv(xi,xj)

# 		#xi = particles
# 		#xj = tr.ones_like(xi)
# 		#xj = xj/tr.norm(xj,dim=-1).unsqueeze(-1)
# 		#ratios  = utils.quaternion_X_times_Y_inv(xj,xi)
# 		#ratios = xi
# 		#xj[:,:,1:] *=-1.
# 		#ratios  = utils.quaternion_prod(xi,xj)

# 		#normalize = tr.norm(ratios,dim=-1).clone().detach()
# 		#ratios  = ratios/normalize.unsqueeze(-1)
# 		return ratios


class QuaternionRelativeMeasureMapWeights(RelativeMeasureMap):
	def __init__(self,edges,grad_type,noise_sampler=None,noise_level=-1.,bernoulli_noise=-1., unfaithfulness=False):
		super(QuaternionRelativeMeasureMapWeights,self).__init__(edges,grad_type)
		self.noise_sampler = noise_sampler
		self.noise_level = noise_level
		self.unfaithfulness = unfaithfulness
		self.bernoulli_noise =  bernoulli_noise
	def forward(self,particles, weights):
		#ratios  = utils.quaternion_prod(xi,xj)
		#normalize = tr.norm(ratios,dim=-1).clone().detach()
		#ratios  = ratios/normalize.unsqueeze(-1)
		ratios,RM_weights = self.compute_ratios(particles,weights)
		if self.noise_level>0.:
			ratios = self.add_noise(ratios)
		if self.unfaithfulness:
			ratios,RM_weights = self.add_unfaithfulness(ratios,RM_weights)

		return ratios,RM_weights

	def compute_ratios(self,particles, weights):
		ratios = []
		RM_weights = []
		i = self.edges[0,:]
		j = self.edges[1,:]
		xi = particles[i,:,:]
		xj = particles[j,:,:]
		
		if self.grad_type=='euclidean':
			ratios = utils.forward_quaternion_X_times_Y_inv(xi,xj)
		elif self.grad_type=='quaternion':
			ratios  = utils.quaternion_X_times_Y_inv(xi,xj)
		#normalize = tr.norm(ratio=-1).clone().detach()
		#ratios  = ratios/normalize.unsqueeze(-1)
		
		#RM_weights = weights[i,:]*weights[j,:]
		N = xi.shape[0]

		RM_weights = weights[0,:].unsqueeze(0).repeat(N,1)
		#ratios = ratios.clone().detach()
		return ratios,RM_weights


	def add_unfaithfulness(self,ratios,RM_weights):
		N,num_particles, _ = ratios.shape
		mask_int =tr.multinomial(RM_weights[0,:],N)
		mask = tr.nn.functional.one_hot(mask_int,num_particles).to(ratios.device)
		mask = mask.type(ratios.dtype)

		ratios = tr.einsum('nki,nk->ni',ratios,mask).unsqueeze(1)
		RM_weights = tr.ones([N,1],dtype=ratios.dtype, device=ratios.device)
		#mask = tr.bernoulli(self.bernoulli_noise*tr.ones([N,num_particles], dtype=ratios.dtype, device=ratios.device))
		return ratios,RM_weights

	def add_noise(self):
		noisy_ratios = add_noise_quaternion(self.noise_sampler,ratios,self.noise_level)
		if self.bernoulli_noise>0.:
			N,num_particles, _ = ratios.shape
			mask = tr.bernoulli(self.bernoulli_noise*tr.ones([N,num_particles], dtype=ratios.dtype, device=ratios.device))
			ratios[mask,:] = noisy_ratios[mask,:]
			return ratios
		else:
			return noisy_ratios

class QuaternionRelativeMeasureMapWeightsProduct(QuaternionRelativeMeasureMapWeights):
	def __init__(self,edges,grad_type,noise_sampler=None,noise_level=-1.,bernoulli_noise=-1., unfaithfulness=False):
		super(QuaternionRelativeMeasureMapWeightsProduct,self).__init__(edges,grad_type,noise_sampler=noise_sampler,noise_level=noise_level,bernoulli_noise=bernoulli_noise, unfaithfulness=unfaithfulness)


	def compute_ratios(self,particles,weights):

		ratios = []
		RM_weights = []
		i = self.edges[0,:]
		j = self.edges[1,:]
		xi = particles[i,:,:]
		xj = particles[j,:,:]
		N,K,_ = xi.shape
		N,L,_ = xj.shape

		if self.grad_type=='euclidean':
			ratios = utils.forward_quaternion_X_times_Y_inv_prod(xi,xj)
		elif self.grad_type=='quaternion':
			ratios = utils.quaternion_X_times_Y_inv_prod(xi,xj)

		ratios = ratios.permute([0,3,1,2]).reshape([N,4,-1]).permute([0,2,1])
		RM_weights = tr.einsum('nk,nl->nkl', weights[i,:],weights[j,:]).reshape([N,-1])

		return ratios,RM_weights


# class QuaternionRelativeMeasureMapProduct(RelativeMeasureMap):
# 	def __init__(self,edges,grad_type):
# 		super(QuaternionRelativeMeasureMapProduct,self).__init__(edges,grad_type)
# 	def forward(self,particles):
# 		ratios = []
# 		i = self.edges[0,:]
# 		j = self.edges[1,:]
# 		xi = particles[i,:,:]
# 		xj = particles[j,:,:]
# 		N,K,_ = xi.shape
# 		N,L,_ = xj.shape
# 		#xj[:,:,1:] = -xj[:,:,1:]
# 		#bb = utils.quaternion_a_inv_times_b(xi,xj)
# 		if self.grad_type=='euclidean':
# 			ratios = utils.forward_quaternion_X_times_Y_inv_prod(xi,xj)
# 		elif self.grad_type=='quaternion':
# 			ratios = utils.quaternion_X_times_Y_inv_prod(xi,xj)

# 		ratios = ratios.permute([0,3,1,2]).reshape([N,4,-1]).permute([0,2,1])

		
# 		return ratios





