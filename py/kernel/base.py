

class BaseKernel(object):
	def __init__(self, D):
		self.D = D 
		self.params =1.
		self.isNull = False
	
	def set_params(self,params):
		raise NotImplementedError()
	def get_params(self):
		raise NotImplementedError()

	def kernel(self, X,Y):

		# Gramm matrix between vectors X and basis
		# Inputs:
		# X 	: N by d matrix of data points
		# basis : M by d matrix of basis points
		# output: N by M matrix
		raise NotImplementedError()

	def derivatives(self,X,Y):
		# Computes the Hadamard product between the gramm matrix of (Y, basis) and the matrix K
		# Input:
		# X 	: N by d matrix of data points
		# Y 	: M by d matrix of basis points		
		# output: N by M matrix   K had_prod Gramm(Y, basis)
		raise NotImplementedError()

	def __add__(self, other):
		if other.D != self.D:
			raise NameError('Dimensions of kernels do not match !')
		else:
			new_kernel = CombinedKernel(self.D, [self, other])
			return new_kernel


class CombinedKernel(BaseKernel):
	def __init__(self,D,  kernels):
		BaseKernel.__init__(self, D)
		self.kernels = kernels

	def kernel(self, X,Y):
		K = 0.
		for kernel in self.kernels:
			K += kernel.kernel(X, Y)
		return K 


	def derivatives(self,X,Y):

		raise NotImplementedError()




