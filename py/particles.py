import  torch as tr
from torch.autograd import Variable
import torch.nn as nn
from torch import autograd
import utils
import numpy as np

class Particles(nn.Module):
	def __init__(self,prior,N, num_particles , noise_level , noise_decay, particle_type='euclidian'):
		super(Particles,self).__init__()
		assert prior.type==particle_type
		self.prior = prior
		self.particle_type= particle_type
		self.num_particles = num_particles
		self.N = N
		self.noise_level = noise_level
		self.noise_decay = noise_decay
		
		self.data = nn.Parameter(prior.sample(N,num_particles)) # N x num_particles x d
		
	def add_noise(self):
		noise = self.prior.sample(self.N,self.num_particles)
		return self.data + self.noise_level*noise 

	def update_noise_level(self):
		self.noise_level *=self.noise_decay

class QuaternionParticles(Particles):
	def __init__(self,prior, N, num_particles, noise_level, noise_decay):
		# particle is a tensor   of shape N x num_paricles x d
		# where d = 4 is the dimension of a normalized quaternion
		#prior = BinghamGenerator(maxNumModes)
		super(QuaternionParticles,self).__init__(prior,N,num_particles, noise_level,noise_decay, particle_type='quaternion')

	def add_noise(self):

		# first sample from prior then scale the rotation by factor level_noise 
		# maybe the sampling can be done more efficiently!!

		noise = self.prior.sample(self.N,self.num_particles)
		angle = tr.acos(noise[:,:,0])
		direction = noise[:,:,1:]/tr.norm(noise[:,:,1:], dim=-1).unsqueeze(-1)
		noise[:,:,0] = tr.cos(self.noise_level * angle)
		noise[:,:,1:] = tr.einsum( 'np,npd ->npd' ,tr.sin(self.noise_level* angle), noise[:,:,1:] )
		
		# compose the rotations

		noisy_data = utils.quaternion_prod(self.data,noise)
		# pos-hoc normalization
		noisy_data = noisy_data/tr.norm(noisy_data, dim=-1).unsqueeze(-1)
		return noisy_data

class RelativeMeasureMap(nn.Module):
	def __init__(self,edges):
		super(RelativeMeasureMap,self).__init__()
		self.edges = edges

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


class QuaternionRelativeMeasureMap(RelativeMeasureMap):
	def __init__(self,edges):
		super(QuaternionRelativeMeasureMap,self).__init__(edges)
	def forward(self,particles):
		ratios = []
		i = self.edges[0,:]
		j = self.edges[1,:]
		xi = particles[i,:,:]
		xj = particles[j,:,:]
		xj[:,:,1:] = -xj[:,:,1:]
		ratios  = utils.quaternion_prod(xi,xj)
		ratios  = ratios/tr.norm(ratios,dim=-1).unsqueeze(-1) 
		return ratios


