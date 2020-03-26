import  torch as tr
from torch.autograd import Variable
import torch.nn as nn
from torch import autograd
import utils
import numpy as np

import sinkhorn_divergence as sd

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
	def __init__(self,prior, N, num_particles,num_edges ,with_weights,with_couplings, product_particles,noise_level , noise_decay, particle_type='euclidian'):
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
		self.with_couplings = with_couplings
		self.data = nn.Parameter(prior.sample(N,num_particles)) # N x num_particles x d
		self.new_thing = False
		if self.product_particles:
			self._weights = (1./np.sqrt(num_particles))*tr.ones([N,num_particles],  dtype=self.data.dtype, device = self.data.device  )
		else:
			self._weights = (1./np.sqrt(num_particles))*tr.ones([num_particles],  dtype=self.data.dtype, device = self.data.device  )
			self._all_weights = tr.ones([N,num_particles],  dtype=self.data.dtype, device = self.data.device )
		if self.new_thing:
			self._weights = (1./num_particles)*tr.ones([num_edges,num_particles*num_particles],  dtype=self.data.dtype, device = self.data.device  )
		if self.with_weights:
			self._weights = nn.Parameter(self._weights)
		if self.product_particles and self.with_couplings:
			self.coupling_strenght =  tr.randn([num_edges,num_particles,num_particles],  dtype=self.data.dtype, device = self.data.device  )
			self.coupling_strenght = nn.Parameter(self.coupling_strenght)
		#self._all_weights = Variable(self._all_weights)
		#self._all_weights.requires_grad = False
	def add_noise(self):
		noise = self.prior.sample(self.N,self.num_particles)
		return self.data + self.noise_level*noise 

	def update_noise_level(self):
		self.noise_level *=self.noise_decay
	def weights(self):
		if self.product_particles and self.with_couplings:
			return  self._weights**2,(self.coupling_strenght**2+10.)
			
		elif self.product_particles:
			return self._weights**2
		else:
			out = self._weights**2
			out = out.unsqueeze(0).repeat(self.N,1)
			return out
class QuaternionParticles(Particles):
	def __init__(self,prior, N, num_particles, num_edges,with_weights,with_couplings,product_particles,noise_level, noise_decay):
		# particle is a tensor   of shape N x num_paricles x d
		# where d = 4 is the dimension of a normalized quaternion
		# prior = BinghamGenerator(maxNumModes)
		super(QuaternionParticles,self).__init__(prior,N,num_particles,num_edges,with_weights,with_couplings,product_particles, noise_level,noise_decay, particle_type='quaternion')

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
			i  = np.int(self.edges[k,0])
			j  = np.int(self.edges[k,1])
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

	def forward(self,particles,weights,edges):
		i = edges[:,0]
		j = edges[:,1]
		xi = particles[i,:,:]
		xj = particles[j,:,:]
		ratios = xi - xj

		#ratios = tr.stack(ratios,dim=0)
		#RM_weights = weights[i,:]*weights[j,:]
		N = xi.shape[0] 
		RM_weights = weights[0,:].unsqueeze(0).repeat(N,1)

		#RM_weights = weights
		#ratios = particles 

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

	def forward(self,particles, weights,edges):
		#ratios  = utils.quaternion_prod(xi,xj)
		#normalize = tr.norm(ratios,dim=-1).clone().detach()
		#ratios  = ratios/normalize.unsqueeze(-1)
		ratios,RM_weights = self.compute_ratios(particles,weights,edges)
		if self.noise_level>0.:
			ratios = self.add_noise(ratios)
		if self.unfaithfulness:
			ratios,RM_weights = self.add_unfaithfulness(ratios,RM_weights)

		return ratios,RM_weights

	def compute_ratios(self,particles, weights,edges):
		ratios = []
		RM_weights = []
		i = edges[:,0]
		j = edges[:,1]
		xi = particles[i,:,:]
		xj = particles[j,:,:]
		
		if self.grad_type=='euclidean':
			ratios = utils.forward_quaternion_X_times_Y_inv(xi,xj)
		elif self.grad_type=='quaternion':
			ratios  = utils.quaternion_X_times_Y_inv(xi,xj)
		#normalize = tr.norm(ratios,dim=-1).clone().detach()
		#ratios  = ratios/normalize.unsqueeze(-1)
		
		#RM_weights = weights[i,:]*weights[j,:]
		N = xi.shape[0]

		RM_weights = weights[0,:].unsqueeze(0).repeat(N,1)
		#ratios = ratios.clone().detach()
		return ratios,RM_weights


	def add_unfaithfulness(self,ratios,RM_weights):
		N,num_particles, _ = ratios.shape
		mask_int =tr.multinomial(RM_weights[0,:],N, replacement=True)
		mask = tr.nn.functional.one_hot(mask_int,num_particles).to(ratios.device)
		mask = mask.type(ratios.dtype)

		ratios = tr.einsum('nki,nk->ni',ratios,mask).unsqueeze(1)
		RM_weights = tr.ones([N,1],dtype=ratios.dtype, device=ratios.device)
		#mask = tr.bernoulli(self.bernoulli_noise*tr.ones([N,num_particles], dtype=ratios.dtype, device=ratios.device))
		return ratios,RM_weights

	def add_noise(self,ratios):
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
		self.test=False
	def forward(self,particles, weights,edges):
		#ratios  = utils.quaternion_prod(xi,xj)
		#normalize = tr.norm(ratios,dim=-1).clone().detach()
		#ratios  = ratios/normalize.unsqueeze(-1)
		ratios,RM_weights,ratios_0 = self.compute_ratios(particles,weights,edges)


	def forward(self,particles,weights,edges):

		ratios = []
		RM_weights = []
		i = edges[:,0]
		j = edges[:,1]
		xi = particles[i,:,:]
		xj = particles[j,:,:]
		N,K,_ = xi.shape
		N,L,_ = xj.shape

		#xj = tr.ones_like(xi)
		#xj = xj/tr.norm(xj,dim=-1).unsqueeze(-1)

		if self.grad_type=='euclidean':
			ratios_0 = utils.forward_quaternion_X_times_Y_inv_prod(xi,xj)
		elif self.grad_type=='quaternion':
			ratios_0 = utils.quaternion_X_times_Y_inv_prod(xi,xj)
		ratios = ratios_0.permute([0,3,1,2]).reshape([N,4,-1]).permute([0,2,1])
		RM_weights = tr.einsum('nk,nl->nkl', weights[i,:],weights[j,:]).reshape([N,-1])
		#RM_weights = weights

		#ratios = particles
		#RM_weights = weights
		if self.test:
			return ratios,RM_weights,ratios_0,xi,xj
		return ratios,RM_weights



class QuaternionRelativeMeasureMapWeightsProductPrior(QuaternionRelativeMeasureMapWeights):
	def __init__(self,edges,num_particles,grad_type,noise_sampler=None,noise_level=-1.,bernoulli_noise=-1., unfaithfulness=False):
		super(QuaternionRelativeMeasureMapWeightsProductPrior,self).__init__(edges,grad_type,noise_sampler=noise_sampler,noise_level=noise_level,bernoulli_noise=bernoulli_noise, unfaithfulness=unfaithfulness)
		self.num_particles = num_particles


	def forward(self,particles, weights,edges):
		#ratios  = utils.quaternion_prod(xi,xj)
		#normalize = tr.norm(ratios,dim=-1).clone().detach()
		#ratios  = ratios/normalize.unsqueeze(-1)
		ratios,RM_weights = self.compute_ratios(particles,weights,edges)
		if self.noise_level>0.:
			ratios = self.add_noise(ratios)

		return ratios,RM_weights

	def compute_ratios(self,particles,weights,edges):

		ratios = []
		RM_weights = []
		i = edges[:,0]
		j = edges[:,1]
		xi = particles[i,:,:]
		xj = particles[j,:,:]
		N,K,_ = xi.shape
		N,L,_ = xj.shape

		#xj = tr.ones_like(xi)
		#xj = xj/tr.norm(xj,dim=-1).unsqueeze(-1)

		#xi,wi,mask_i_0 = self.sample_particles(xi,weights[i,:])
		#xj,wj,mask_j_0 = self.sample_particles(xj,weights[j,:])

		#ratios = tr.einsum('nki,nk->ni',ratios,mask).unsqueeze(1)
		#RM_weights = tr.ones([N,1],dtype=ratios.dtype, device=ratios.device)

		if self.grad_type=='euclidean':
			ratios = utils.forward_quaternion_X_times_Y_inv_prod(xi,xj)
		elif self.grad_type=='quaternion':
			ratios = utils.quaternion_X_times_Y_inv_prod(xi,xj)

		ratios = ratios.permute([0,3,1,2]).reshape([N,4,-1]).permute([0,2,1])
		#RM_weights = tr.einsum('nk,nl->nkl', wi,wj).reshape([N,-1])
		#RM_weights = (1./ratios.shape[1])*tr.ones([N,ratios.shape[1]],dtype=ratios.dtype, device=ratios.device)
		RM_weights = tr.einsum('nk,nl->nkl', weights[i,:],weights[j,:]).reshape([N,-1])
		#RM_weights = tr.einsum('nk,nl->nkl', wi,wj).reshape([N,-1])
		#ratios = particles
		#RM_weights = weights

		return ratios,RM_weights


	def sample_particles(self,particles,weights):
		mask_i_0 =tr.multinomial(weights,self.num_particles, replacement=True)
		mask_i = tr.nn.functional.one_hot(mask_i_0,particles.shape[1]).to(particles.device)
		mask_i = mask_i.type(particles.dtype)
		x = tr.einsum('nki,nlk->nli',particles,mask_i)
		w = tr.einsum('nk,nlk->nl',weights,mask_i)
		return x,w,mask_i_0

	def add_noise(self,ratios):
		noisy_ratios = add_noise_quaternion(self.noise_sampler,ratios,self.noise_level)
		if self.bernoulli_noise>0.:
			N,num_particles, _ = ratios.shape
			mask = tr.bernoulli(self.bernoulli_noise*tr.ones([N,num_particles], dtype=ratios.dtype, device=ratios.device))
			ratios[mask,:] = noisy_ratios[mask,:]
			return ratios
		else:
			return noisy_ratios
	def add_unfaithfulness(self,ratios,RM_weights):
		N,num_particles, _ = ratios.shape
		mask_int =tr.multinomial(RM_weights[0,:],N, replacement=True)
		mask = tr.nn.functional.one_hot(mask_int,num_particles).to(ratios.device)
		mask = mask.type(ratios.dtype)

		ratios = tr.einsum('nki,nk->ni',ratios,mask).unsqueeze(1)
		RM_weights = tr.ones([N,1],dtype=ratios.dtype, device=ratios.device)
		#mask = tr.bernoulli(self.bernoulli_noise*tr.ones([N,num_particles], dtype=ratios.dtype, device=ratios.device))
		return ratios,RM_weights

class QuaternionRelativeMeasureMapWeightsCouplings(QuaternionRelativeMeasureMapWeights):
	def __init__(self,edges,grad_type,noise_sampler=None,noise_level=-1.,bernoulli_noise=-1., unfaithfulness=False):
		super(QuaternionRelativeMeasureMapWeightsCouplings,self).__init__(edges,grad_type,noise_sampler=noise_sampler,noise_level=noise_level,bernoulli_noise=bernoulli_noise, unfaithfulness=unfaithfulness)
		self.subsample = False
		self.a_y = 0.
		self.b_x = 0.
	def compute_ratios(self,particles,couplings,edges):
		weights,coupling_strenght = couplings

		ratios = []
		RM_weights = []
		i = edges[:,0]
		j = edges[:,1]
		xi = particles[i,:,:]
		xj = particles[j,:,:]
		N,K,_ = xi.shape
		N,L,_ = xj.shape

		#xj = tr.ones_like(xi)
		#xj = xj/tr.norm(xj,dim=-1).unsqueeze(-1)

		A = self.coupling(weights[i,:],weights[j,:],coupling_strenght).reshape([N,-1])
		A = A/tr.sum(A,dim=-1).unsqueeze(-1).detach()
		if self.subsample:
			#mask =tr.multinomial(A,particles.shape[1], replacement=True)
			mask = tr.multinomial(tr.ones_like(A),particles.shape[1], replacement=True)
			mask_i = mask/K
			mask_j = mask-mask_i*K

			mask_i = tr.nn.functional.one_hot(mask_i,particles.shape[1]).to(particles.device)
			mask_i = mask_i.type(particles.dtype)
			xi = tr.einsum('nki,nlk->nli',xi,mask_i)

			mask_j = tr.nn.functional.one_hot(mask_j,particles.shape[1]).to(particles.device)
			mask_j = mask_j.type(particles.dtype)
			xj = tr.einsum('nki,nlk->nli',xj,mask_j)

			mask = tr.nn.functional.one_hot(mask,A.shape[1]).to(particles.device)
			mask = mask.type(particles.dtype)
			A = tr.einsum('nk,nlk->nl',A,mask)
			#w = tr.einsum('nk,nlk->nl',weights,mask_i)

			if self.grad_type=='euclidean':
				ratios = utils.forward_quaternion_X_times_Y_inv(xi,xj)
			elif self.grad_type=='quaternion':
				ratios = utils.quaternion_X_times_Y_inv(xi,xj)

			#ratios = ratios.permute([0,3,1,2]).reshape([N,4,-1]).permute([0,2,1])
			#RM_weights = self.coupling(weights[i,:],weights[j,:],coupling_strenght).reshape([N,-1])
			RM_weights = A/tr.sum(A,dim=-1).unsqueeze(-1).detach()
		else:
			if self.grad_type=='euclidean':
				ratios = utils.forward_quaternion_X_times_Y_inv_prod(xi,xj)
			elif self.grad_type=='quaternion':
				ratios = utils.quaternion_X_times_Y_inv_prod(xi,xj)			
			ratios = ratios.permute([0,3,1,2]).reshape([N,4,-1]).permute([0,2,1])
			RM_weights = A/tr.sum(A,dim=-1).unsqueeze(-1).detach()
		#RM_weights =  A
		#ratios = particles
		#RM_weights = weights

		return ratios,RM_weights

	def coupling(self,weights_i,weights_j,coupling_strenght):
		# weights: E x P
		# coupling_strenght:   E x P x P
		# returns a matrix of shape E x P x P containing the couplings between particles
		scaling = 0.5
		blur = .0000001
		ε_s = epsilon_schedule( coupling_strenght, blur, scaling )
		#coupling_strenght = coupling_strenght[:,:,1:3]
		perm_coupling = coupling_strenght.permute(0,2,1)
		log_w_i = sd.log_weights(weights_i)
		#weights_j = weights_j[:,1:3]
		#weights_j = weights_j/tr.norm(weights_j,dim=-1).unsqueeze(-1)
		log_w_j = sd.log_weights(weights_j)
		


		_, _, self.a_y, self.b_x = sd.sinkhorn_loop( softmin_tensorized, log_w_i, log_w_j, None, None, coupling_strenght, perm_coupling , ε_s, None, debias=False , b_x_0= self.b_x, a_y_0=self.a_y)

		A = (1./blur)*(self.a_y.unsqueeze(-2)+self.b_x.unsqueeze(-1) - coupling_strenght)
		A += log_w_i.unsqueeze(-1) + log_w_j.unsqueeze(-2) 
		max_A,_= tr.max(A,dim=-1)
		max_A,_ = tr.max(max_A,dim=-1)
		A -= max_A.unsqueeze(-1).unsqueeze(-1)
		
		A = tr.exp(A)
		A = A/tr.sum(tr.sum(A,dim=-1),dim=-1).unsqueeze(-1).unsqueeze(-1)
		#out = tr.einsum('ni,nij->nij',a_y,tr.exp(-blur*coupling_strenght))
		#out = tr.einsum('nj,nij->nij',b_x,out)
		err = tr.norm( weights_i - tr.sum(A,dim=-1)  ) + tr.norm(weights_j - tr.sum(A,dim=-2))
		print('error: ' +str(err))
		return A

def softmin_tensorized(ε, C, f):
	B = C.shape[0]
	return - ε * ( f.view(B,1,-1) - C/ε ).logsumexp(2).view(B,-1)

def softmin_tensorized2(ε, C, log_a,f):
	B = C.shape[0]
	return ε * ( 0.5+log_a -( f.unsqueeze(-1) - C/ε ).logsumexp(1))



def epsilon_schedule(coupling_strenght, blur, scaling):
	diameter = max(tr.max(coupling_strenght).item(),1.)

	ε_s = [ diameter ] + [ np.exp(e) for e in np.arange(np.log(diameter), np.log(blur), np.log(scaling)) ] + [ blur]
	#ε_s = [ diameter ] + [ np.exp(e) for e in np.arange(np.log(diameter), np.log(blur), np.log(scaling)) ] + [ blur]
	return ε_s

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





