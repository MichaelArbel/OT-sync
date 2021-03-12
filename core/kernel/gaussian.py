import torch as tr
from kernel.base import BaseKernel


from utils import pow_10, FDTYPE, DEVICE

import utils

class Exp(BaseKernel):
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
		raise NotImplementedError()

	def _kernel(self,log_sigma,X,Y):

		sigma = pow_10(log_sigma,dtype=self.dtype,device=self.device)
		dist = self._dist( X, Y)
		return  tr.exp(-dist/sigma)

class Gaussian(Exp):

	def __init__(self, D,  log_sigma, particles_type='euclidian', dtype = FDTYPE, device = DEVICE):
		Exp.__init__(self, D, log_sigma, particles_type=particles_type, dtype = dtype, device = device)
		self.kernel_type = 'squared_euclidean'
	def _dist(self,X, Y):
		tmp = (X.unsqueeze(-2) - Y.unsqueeze(-3))**2
		dist =  tr.sum(tmp,dim=-1)
		return dist
class ExpQuaternionGeodesicDist(Exp):
	def __init__(self, D,  log_sigma, particles_type='euclidian', dtype = FDTYPE, device = DEVICE):
		Exp.__init__(self, D, log_sigma, particles_type=particles_type, dtype = dtype, device = device)
		self.kernel_type = 'quaternion'
	def _dist(self,X, Y):
			return utils.quaternion_geodesic_distance(X,Y)

class ExpPowerQuaternionGeodesicDist(Exp):
	def __init__(self,power, D,  log_sigma, particles_type='euclidian', dtype = FDTYPE, device = DEVICE):
		Exp.__init__(self, D, log_sigma, particles_type=particles_type, dtype = dtype, device = device)
		self.power = power
		self.kernel_type = 'power_quaternion'
	def _dist(self,X, Y):
			return utils.quaternion_geodesic_distance(X,Y)**self.power















