import torch as tr
from kernel.base import BaseKernel
from utils import pow_10, FDTYPE, DEVICE
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

	def _square_dist(self,X, Y):
		if self.particles_type=='euclidian':
			n_x,d = X.shape
			n_y,d = Y.shape
			dist =  -2*tr.einsum('mr,nr->mn',X,Y) + tr.sum(X**2,1).unsqueeze(-1).repeat(1,n_y) +  tr.sum(Y**2,1).unsqueeze(0).repeat(n_x,1) #  tr.einsum('m,n->mn', tr.ones([ n_x],dtype=self.dtype, device = self.device),tr.sum(Y**2,1)) 
			return dist
		elif self.particles_type=='quaternion':
			X = X.unsqueeze(1) - 0.*Y.unsqueeze(2)
			Y = Y.unsqueeze(2) - 0.*X
			return utils.quaternion_geodesic_distance(X,Y)**2
		else:
			raise NotImplementedError()

	def _kernel(self,log_sigma,X,Y):

		sigma = pow_10(log_sigma,dtype=self.dtype,device=self.device)
		tmp = self._square_dist( X, Y)
		dist = tr.max(tmp,tr.zeros_like(tmp))
		if len(dist.shape)>2:
			return  tr.sum(tr.exp(-0.5*dist/sigma),dim=0)
		else:
			return  tr.exp(-0.5*dist/sigma)



























