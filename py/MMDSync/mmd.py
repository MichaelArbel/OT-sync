import  torch as tr
from torch.autograd import Variable
import torch.nn as nn
from torch import autograd





class MMD(nn.Module):
	def __init__(self,kernel, particles, rm_map, with_noise):
		super(MMD,self).__init__()
		self.kernel = kernel
		self.particles = particles
		self.rm_map = rm_map
		self.mmd2 =  mmd2_func.apply
		self.with_noise = with_noise
		
	def forward(self, true_data):
		if self.with_noise:
			noisy_data = self.rm_map(self.particles.add_noise())
		else:
			noisy_data = self.rm_map(self.particles.data)

		fake_data =  self.rm_map(self.particles.data.clone().detach())
		mmd2_val = self.mmd2(self.kernel,true_data,fake_data,noisy_data)
		return mmd2_val



class MMD_weighted(nn.Module):
	def __init__(self,kernel, particles, rm_map, with_noise):
		super(MMD_weighted,self).__init__()
		self.kernel = kernel
		self.particles = particles
		self.rm_map = rm_map
		self.mmd2 =  mmd2_weights_func.apply
		self.with_noise = with_noise
		
	def forward(self, true_data,true_weights):
		if self.with_noise:
			noisy_data, weights = self.rm_map(self.particles.add_noise(),self.particles.weights())
		else:
			noisy_data, weights = self.rm_map(self.particles.data,self.particles.weights())

		fake_data,fake_weights =  self.rm_map(self.particles.data.clone().detach(),self.particles.weights().clone().detach())
		mmd2_val = self.mmd2(self.kernel,true_data,true_weights,fake_data,fake_weights,noisy_data, weights)
		return mmd2_val



class mmd2_func(tr.autograd.Function):

	@staticmethod
	def forward(ctx, kernel,true_data,fake_data,noisy_data):

		with  tr.enable_grad():
			gram_XY = kernel.kernel(true_data,noisy_data)
			gram_XX = kernel.kernel(true_data, true_data)
			gram_YY = kernel.kernel(fake_data,noisy_data)
			gram_YY = kernel.kernel(noisy_data,noisy_data)
			#gram_YY_t = kernel.kernel(fake_data,fake_data)
			#gram_XY_t = kernel.kernel(true_data,fake_data)
			N_cameras,N_y,N_z = gram_YY.shape
			#mmd2_for_grad =  N_z*(tr.mean(gram_YY) - tr.mean(gram_XY))
			term_YY = N_cameras*tr.mean(gram_YY)
			term_XY = N_cameras*tr.mean(gram_XY)
			term_XX = N_cameras*tr.mean(gram_XX)
			#mmd2_for_grad =  N_z*(term_YY - term_XY)
			mmd2 =  term_XX + term_YY -2.*term_XY
			mmd2_for_grad =  N_z*mmd2
			##### warning this is dangerous
			#mmd2_for_grad =  N_z*(tr.mean(gram_XY))
			#mmd2 = mmd2_for_grad
			

		ctx.save_for_backward(mmd2_for_grad,mmd2,noisy_data)

		return 0.5*mmd2

	@staticmethod
	def backward(ctx, grad_output):

		mmd2_for_grad,mmd2, noisy_data = ctx.saved_tensors
		with  tr.enable_grad():
			gradients = autograd.grad(outputs=mmd2_for_grad, inputs=noisy_data,
					  	grad_outputs=grad_output,
					 	create_graph=True, only_inputs=True)[0]
		#return aa
		return None, None, None, gradients

class mmd2_weights_func(tr.autograd.Function):

	@staticmethod
	def forward(ctx, kernel,true_data,true_weights,fake_data,fake_weights, noisy_data, weights):

		with  tr.enable_grad():
			gram_XY = kernel.kernel(true_data,noisy_data)
			gram_XX = kernel.kernel(true_data,true_data)
			gram_YY = kernel.kernel(fake_data,noisy_data)
			#gram_YY = kernel.kernel(noisy_data,noisy_data)
			#gram_YY_t = kernel.kernel(fake_data,fake_data)
			#gram_XY_t = kernel.kernel(true_data,fake_data)
			N_cameras,N_x, _ = gram_XX.shape
			_,N_y, N_z = gram_YY.shape
			#mmd2_for_grad =  N_z*(tr.mean(gram_YY) - tr.mean(gram_XY))
			term_YY = tr.einsum('nkl,nk,nl->' , gram_YY, fake_weights,weights)
			term_XY = tr.einsum('nkl,nk,nl->' ,gram_XY, true_weights,weights)
			term_XX = tr.einsum('nkl,nk,nl->' ,gram_XX, true_weights,true_weights)
			mmd2_for_grad =  N_z*(term_YY - term_XY)
			
			mmd2 =  term_XX + term_YY -2.*term_XY
			#mmd2_for_grad =  N_z*mmd2
			##### warning this is dangerous
			#mmd2_for_grad =  N_z*(tr.mean(gram_XY))
			#mmd2 = mmd2_for_grad
			

		ctx.save_for_backward(mmd2_for_grad,mmd2,noisy_data,weights)

		return 0.5*mmd2

	@staticmethod
	def backward(ctx, grad_output):

		mmd2_for_grad,mmd2, noisy_data, weights = ctx.saved_tensors
		with  tr.enable_grad():
			if noisy_data.requires_grad and weights.requires_grad:
				gradients = autograd.grad(outputs=mmd2_for_grad, inputs=[noisy_data,weights],
						  	grad_outputs=grad_output,
						 	create_graph=True, only_inputs=True)
				return None, None,None, None, None, gradients[0],gradients[1] 
			elif not noisy_data.requires_grad and weights.requires_grad:
				gradients = autograd.grad(outputs=mmd2_for_grad, inputs=[weights],
						  	grad_outputs=grad_output,
						 	create_graph=True, only_inputs=True)
				return None, None,None, None, None, None,gradients[0]
			elif  noisy_data.requires_grad and not weights.requires_grad:
				gradients = autograd.grad(outputs=mmd2_for_grad, inputs=[noisy_data],
						  	grad_outputs=grad_output,
						 	create_graph=True, only_inputs=True)
				return None, None, None, None, None, gradients[0],None
