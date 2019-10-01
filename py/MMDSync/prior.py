import  torch as tr
from torch.autograd import Variable
import torch.nn as nn
from torch import autograd
from utils import *
import os
import numpy as np



class Prior(object):
	def __init__(self, name = 'prior'):
		self.name = name
		self.type = 'euclidian' 
	def sample(self,N, num_particles):
		raise NotImplementedError()

class MixtureGaussianPrior(Prior):
	def __init__(self,maxNumModes):
		super(MixtureGaussianPrior,self).__init__(name='mixture_gaussians')
		self.maxNumModes = maxNumModes
	def sample(self,N,num_particles):
		Xs = []
		for i in range(0, N):
			numModes = np.random.randint(self.maxNumModes)+1
			mus = np.random.randn(numModes)
			Vars = np.random.rand(numModes)*0.3
			covs = np.diag(Vars)
			X = np.ones((numParticles))
			k = 0
			numPtsPerMode = np.int(numParticles/self.maxNumModes)
			for j in range(self.maxNumModes):
				xcur = np.random.normal(loc=0.0, scale=1.0,size=numPtsPerMode)

				X[k:k+numPtsPerMode] = xcur

				k=k+numPtsPerMode
			Xs.append(X)
		Xs = tr.tensor(np.array(Xs).unsqueeze(-1))
		return Xs

class GaussianQuaternionPrior(Prior):
	def __init__(self):
		super(GaussianQuaternionPrior,self).__init__(name='gaussian')
		self.type= 'quaternion'
	def sample(self,N,num_particles):
		Xs = np.random.randn(N,num_particles,4)
		Xs = tr.tensor(Xs)
		Xs = Xs/tr.norm(Xs,dim=-1).unsqueeze(-1)

		return Xs


