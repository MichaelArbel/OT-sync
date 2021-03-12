import  torch as tr
from torch.autograd import Variable
import torch.nn as nn
from torch import autograd
from utils import *
import os
import numpy as np



class Prior(object):
	def __init__(self, dtype,device, name = 'prior'):
		self.name = name
		self.type = 'euclidian'
		self.dtype = dtype
		self.device = device
	def sample(self,N, num_particles):
		raise NotImplementedError()

class MixtureGaussianPrior(Prior):
	def __init__(self,maxNumModes,dtype,device):
		super(MixtureGaussianPrior,self).__init__(dtype,device,name='mixture_gaussians')
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
		Xs = tr.tensor(np.array(Xs).unsqueeze(-1),dtype=self.dtype, device=self.device)
		Xs[:,:,0] = tr.abs(Xs[:,:,0])
		return Xs

class GaussianQuaternionPrior(Prior):
	def __init__(self,dtype,device):
		super(GaussianQuaternionPrior,self).__init__(dtype,device,name='gaussian')
		self.type= 'quaternion'
	def sample(self,N,num_particles):
		Xs = np.random.randn(N,num_particles,4)
		Xs = tr.tensor(Xs,dtype=self.dtype, device=self.device)
		Xs[:,:,0] = tr.abs(Xs[:,:,0])

		Xs = Xs/tr.norm(Xs,dim=-1).unsqueeze(-1)

		return Xs

class GaussianPrior(Prior):
	def __init__(self,dtype,device):
		super(GaussianPrior,self).__init__(dtype,device,name='gaussian')
		self.type= 'euclidian'
	def sample(self,N,num_particles):
		Xs = np.random.randn(N,num_particles,4)
		Xs = tr.tensor(Xs,dtype=self.dtype, device=self.device)
		#Xs[:,:,0] = tr.abs(Xs[:,:,0])

		#Xs = Xs/tr.norm(Xs,dim=-1).unsqueeze(-1)

		return Xs

# class MixtureBingham(Prior):
# 	def __init__(self,dtype, device):
# 		super(MixtureBingham,self).__init__(dtype,device,name='mixture_bingham')
# 		self.type = 'mixture_bingham'
# 		import matlab.engine
# 		self.eng = matlab.engine.start_matlab()

# 	def sample(self, N,num_particles):
# 		X_s = np.random.randn(N,num_particles,4)
# 		bingham = db.get_bingham(eng, distributions, GT=None, precision=quality)  # without ground truth
		
# 		n = matlab.double([n])
# 		B = [V,Z,F]
# 		Y = eng.bingham_sample(V,Z,n)

# 		Xs = tr.tensor(Xs,dtype=self.dtype, device=self.device)
# 		#Xs[:,:,0] = tr.abs(Xs[:,:,0])

# 		#Xs = Xs/tr.norm(Xs,dim=-1).unsqueeze(-1)

# 		return Xs



# class Anular(Prior):
# 	def __init__(self, maxNumModes, dtype, device):
# 		super(Anular,self).__init__(dtype,device,name='anular')
# 		self.maxNumModes = maxNumModes
# 	def sample(self, N, num_particles):



#     # generate emprical prior distributions for each node
#     for i in range(0, N):
    
#         numModes = np.random.randint(maxNumModes)+1
#         mus = np.random.randn(numModes)
#         kappa = 0.3*np.pi
#         kappas = kappa*np.ones((int(numModes),1))  # circular dispersion
        
#         Y = np.ones((numParticles))/numParticles
#         X = np.ones((numParticles))
        
#         k = 0
#         numPtsPerMode = np.int(numParticles/maxNumModes)
#         for j in range(numModes):
#             #print('j ',str(j), ' , ', len(mus))            
#             mu = mus[j]
#             xcur = np.random.vonmises(mu, kappa, size=numPtsPerMode)
#             X[k:k+numPtsPerMode] = xcur
#             k=k+numPtsPerMode
        
#         Xs.append(X)

#     return synthData






