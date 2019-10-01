import  torch as tr
from torch.autograd import Variable
import torch.nn as nn
from torch import autograd





class MMD_noise_injection(nn.Module):
	def __init__(self,kernel, particles, rm_map, with_noise):
		super(MMD_noise_injection,self).__init__()
		self.kernel = kernel
		self.particles = particles
		self.rm_map = rm_map
		self.mmd2_noise =  mmd2_noise_injection.apply
		self.mmd2 = mmd2.apply
		self.with_noise = with_noise
		
	def forward(self, true_data):
		if self.with_noise:
			noisy_data = self.rm_map(self.particles.add_noise()) 
			fake_data =  self.rm_map(self.particles.data.clone().detach())
			#mmd2_val = 0.
			#for e in range(len(noisy_data)):
			mmd2_val = self.mmd2_noise(self.kernel,true_data,fake_data,noisy_data)
		else:
			fake_data =  self.rm_map(self.particles.data)
			#mmd2_val = 0.
			#for e in range(len(noisy_data)):
			mmd2_val = self.mmd2(self.kernel,true_data,fake_data)
		return mmd2_val


class mmd2(tr.autograd.Function):

	@staticmethod
	def forward(ctx, kernel,true_data,fake_data):

		with  tr.enable_grad():
			gram_XY = kernel.kernel(true_data,fake_data)
			gram_XX = kernel.kernel(true_data, true_data)
			gram_YY = kernel.kernel(fake_data,fake_data)
			N_x, _ = gram_XX.shape
			N_y, _ = gram_YY.shape
			if N_x >1:
				mmd2 = (1./(N_x*(N_x-1)))*(tr.sum(gram_XX)-tr.trace(gram_XX)) \
					+ (1./(N_y*(N_y-1)))*(tr.sum(gram_YY)-tr.trace(gram_YY)) \
					- 2.* tr.mean(gram_XY)
			else: 
				mmd2 = tr.sum(gram_XX) \
					+ (1./(N_y*(N_y-1)))*(tr.sum(gram_YY)-tr.trace(gram_YY)) \
					- 2.* tr.mean(gram_XY)				
			mmd2_for_grad = 0.5*N_y*mmd2.clamp(min=0)

		ctx.save_for_backward(mmd2_for_grad,fake_data)

		return 0.5*mmd2.clamp(min=0)

	@staticmethod
	def backward(ctx, grad_output):
		mmd2_for_grad, fake_data = ctx.saved_tensors
		with  tr.enable_grad():
			gradients = autograd.grad(outputs=mmd2_for_grad, inputs=fake_data,
					  	grad_outputs=grad_output,
					 	create_graph=True, only_inputs=True)[0] 
				
		return None, None, gradients




class mmd2_noise_injection(tr.autograd.Function):

	@staticmethod
	def forward(ctx, kernel,true_data,fake_data,noisy_data):

		with  tr.enable_grad():
			gram_XY = kernel.kernel(true_data,noisy_data)
			gram_XX = kernel.kernel(true_data, true_data)
			gram_YY = kernel.kernel(fake_data,noisy_data)
			gram_YY_t = kernel.kernel(fake_data,fake_data)
			gram_XY_t = kernel.kernel(true_data,fake_data)
			N_x, _ = gram_XX.shape
			N_y, N_z = gram_YY.shape
			mmd2_for_grad =  N_z*( 1./(N_y*(N_y-1))*(tr.sum(gram_YY)-tr.trace(gram_YY))  - tr.mean(gram_XY))
			if N_x>1:
				mmd2 = (1./(N_x*(N_x-1)))*(tr.sum(gram_XX)-tr.trace(gram_XX)) \
					+ (1./(N_y*(N_y-1)))*(tr.sum(gram_YY_t)-tr.trace(gram_YY_t)) \
					- 2.* tr.mean(gram_XY_t)
			else:
				mmd2 = tr.sum(gram_XX) \
					+ (1./(N_y*(N_y-1)))*(tr.sum(gram_YY_t)-tr.trace(gram_YY_t)) \
					- 2.* tr.mean(gram_XY_t)

		ctx.save_for_backward(mmd2_for_grad,mmd2,noisy_data)

		return 0.5*mmd2.clamp(min=0)

	@staticmethod
	def backward(ctx, grad_output):

		mmd2_for_grad,mmd2, noisy_data = ctx.saved_tensors
		with  tr.enable_grad():
			gradients = autograd.grad(outputs=mmd2_for_grad, inputs=noisy_data,
					  	grad_outputs=grad_output,
					 	create_graph=True, only_inputs=True)[0] 
			gradients = gradients*(mmd2>0)
		return None, None, None, gradients

