import torch as tr
from kernel.base import BaseKernel


from utils import pow_10, FDTYPE, DEVICE, quaternion_geodesic_distance

import utils

class Gaussian(BaseKernel):
	def __init__(self, D,  log_sigma, particles_type='euclidian', dtype = FDTYPE, device = DEVICE):
		BaseKernel.__init__(self, D)
		self.particles_type = particles_type
		self.params  = log_sigma
		self.dtype = dtype
		self.device = device
	def get_exp_params(self):
		return pow_10(self.params,dtype=self.dtype,device=self.device)
	def update_params(self,log_sigma):
		self.params = log_sigma

	def square_dist(self, X, Y):

		return self._square_dist( X, Y)

	def kernel(self, X,Y):
		return self._kernel(self.params,X, Y)

	def derivatives(self, X,Y):
		return self._derivatives(self.params,X,Y)

	def _dist(self,X, Y):
		tmp = (X.unsqueeze(-2) - Y.unsqueeze(-3))**2
		dist =  tr.sum(tmp,dim=-1)
		return dist

	def _kernel(self,log_sigma,X,Y):

		sigma = pow_10(log_sigma,dtype=self.dtype,device=self.device)
		dist = self._dist( X, Y)
		return  tr.exp(-dist/sigma)

class LaplaceQuaternionGeodesicDist(Gaussian):
	def __init__(self, D,  log_sigma, particles_type='euclidian', dtype = FDTYPE, device = DEVICE):
		Gaussian.__init__(self, D, log_sigma, particles_type=particles_type, dtype = dtype, device = device)
	def _dist(self,X, Y):
			return quaternion_geodesic_distance(X,Y)



class GaussianQuaternionGeodesicDist(Gaussian):
	def __init__(self, D,  log_sigma, particles_type='euclidian', dtype = FDTYPE, device = DEVICE):
		Gaussian.__init__(self, D, log_sigma, particles_type=particles_type, dtype = dtype, device = device)
	def _dist(self,X, Y):
			return quaternion_geodesic_distance(X,Y)**2















