import torch as tr
from kernel.base import BaseKernel
from Utils import pow_10, FDTYPE, DEVICE



class Gaussian(BaseKernel):
	def __init__(self, D,  log_sigma, dtype = FDTYPE, device = DEVICE):
		BaseKernel.__init__(self, D)
		self.params  = log_sigma
		self.dtype = dtype
		self.device = device

	# def exp_params(self,sigma):
	# 	return pow_10(sigma)
	def get_exp_params(self):
		return pow_10(self.params,dtype=self.dtype,device=self.device)
	def update_params(self,log_sigma):
		self.params = log_sigma
	# def set_params(self,params):
	# 	# stores the inverse of the bandwidth of the kernel
	# 	if isinstance(sigma, float):
 #            self.params  = tr.from_numpy(sigma)
 #        elif type(sigma)==tr.Tensor:
 #            self.params = sigma
 #        else:
 #            raise NameError("sigma should be a float or tf.Tensor")
	# 	self.params = params

	# def get_params(self):
	# 	# returns the bandwidth of the gaussian kernel
	# 	return self.params


	def square_dist(self, X, Y):
		# Squared distance matrix of pariwise elements in X and basis
		# Inputs:
		# X 	: N by d matrix of data points
		# basis : M by d matrix of basis points
		# output: N by M matrix

		return self._square_dist( X, Y)

	def kernel(self, X,Y):

		# Gramm matrix between vectors X and basis
		# Inputs:
		# X 	: N by d matrix of data points
		# basis : M by d matrix of basis points
		# output: N by M matrix

		return self._kernel(self.params,X, Y)

	def derivatives(self, X,Y):
		return self._derivatives(self.params,X,Y)



# Private functions 

	def _square_dist(self,X, Y):
		n_x,d = X.shape
		n_y,d = Y.shape
#		dist = -2*tr.einsum('mr,nr->mn',X,Y) + tr.einsum('m,n->mn',tr.sum(X**2,1), tr.ones([ n_y],dtype=self.dtype, device = self.device)) +  tr.einsum('m,n->mn', tr.ones([ n_x],dtype=self.dtype, device = self.device),tr.sum(Y**2,1)) 
		dist = -2*tr.einsum('mr,nr->mn',X,Y) + tr.sum(X**2,1).unsqueeze(-1).repeat(1,n_y) +  tr.sum(Y**2,1).unsqueeze(0).repeat(n_x,1) #  tr.einsum('m,n->mn', tr.ones([ n_x],dtype=self.dtype, device = self.device),tr.sum(Y**2,1)) 

		return dist 

	def _kernel(self,log_sigma,X,Y):

		sigma = pow_10(log_sigma,dtype=self.dtype,device=self.device)
		tmp = self._square_dist( X, Y)
		dist = tr.max(tmp,tr.zeros_like(tmp))
		return  tr.exp(-0.5*dist/sigma)

	def _derivatives(self,log_sigma,X,Y):
		# X : [M,T]
		# Y : [N,R]

		# dkdxdy ,   dkdxdy2  = [M,N,T,R]  
		# dkdy2 = [M,N,R]
		# dkdY = [M,N,R]
		# dkdx = [M,N,T]
		# gram =  [M,N]
		N,d = X.shape
		assert d==1 
		sigma = pow_10(log_sigma,dtype=self.dtype,device=self.device)
		gram = self._kernel(log_sigma,X, Y)

		D = (X.unsqueeze(1) - Y.unsqueeze(0))/sigma
		 
		I  = tr.ones( D.shape[-1],dtype=self.dtype, device = self.device)/sigma

		dkdy = tr.einsum('mn,mnr->mnr', gram,D)
		dkdx = -dkdy


		D2 = D**2
		dkdy2 = D2-I

		dkdy2 = tr.einsum('mn,mnr->mnr', gram,dkdy2)

		
		D2 = tr.einsum('mnt,mnr->mntr', D, D)

		I  = tr.eye( D.shape[-1],dtype=self.dtype, device = self.device)/sigma
		dkdxdy = I - D2
		dkdxdy = tr.einsum('mn, mntr->mntr', gram, dkdxdy)

		h =  1./sigma - D**2


		hD = tr.einsum('mnt,mnr->mntr',D,h)

		hD2 = 2.*tr.einsum('mnr,tr->mntr',D,I)

		dkdxdy2 = hD2 + hD

		dkdxdy2 = tr.einsum('mn,mntr->mntr', gram, dkdxdy2)


		#dkdxdy = tr.einsum('mntr->mn',dkdxdy)
		#dkdxdy2 = tr.einsum('mntr->mn',dkdxdy2)
		#dkdy2 = tr.einsum('mnr->mn', dkdy2)
		#dkdy = tr.einsum('mnr->mn', dkdy)
		#dkdx = tr.einsum('mnr->mn', dkdx)

		return dkdxdy, dkdxdy2, dkdx, dkdy, dkdy2, gram





























