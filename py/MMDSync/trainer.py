import os
import time 
import numpy as np
import sys



import torch
import torch.optim as optim
import torch.optim as optim

from tensorboardX import SummaryWriter

import utils

import mmd
import particles
import prior
import optimizers
from kernel.gaussian import Gaussian

class Trainer(object):
	def __init__(self,args):
		self.args = args
		self.device = 'cuda:'+str(args.device) if torch.cuda.is_available() and args.device>-1 else 'cpu'
		self.dtype = get_dtype(args)
		self.log_dir = os.path.join(args.log_dir, args.log_name+ '_model_' + args.model+'_loss_' + args.loss + '_particles_' + args.particles_type )

		if not os.path.isdir(self.log_dir):
			os.mkdir(self.log_dir)
		
		if args.log_in_file:
			self.log_file = open(os.path.join(self.log_dir, 'log.txt'), 'w', buffering=1)
			sys.stdout = self.log_file

		print('Creating writer')
		self.writer = SummaryWriter(self.log_dir)

		print('==> Building model..')
		self.build_model()


	def build_model(self):
		torch.manual_seed(self.args.seed)
		self.edges = get_edges(self.args)
		self.RM_map = get_rm_map(self.args,self.edges)
		self.prior = get_prior(self.args)
		self.true_RM = self.get_true_rm()
		self.particles = get_particles(self.args, self.prior)
		self.kernel = get_kernel(self.args,self.dtype, self.device)	
		self.loss = self.get_loss()
		self.optimizer = self.get_optimizer(self.args.lr)
		self.scheduler = get_scheduler(self.args, self.optimizer)

	def get_loss(self):
		if self.args.loss=='mmd_noise_injection':
			return mmd.MMD_noise_injection(self.kernel, self.particles,self.RM_map, with_noise = True)
		elif self.args.loss=='mmd':
			return mmd.MMD_noise_injection(self.kernel, self.particles,self.RM_map, with_noise = False)

		else:
			raise NotImplementedError()
	def get_optimizer(self,lr):
		if self.args.particles_type=='euclidian':

			if self.args.optimizer=='SGD':
				return optim.SGD(self.particles.parameters(), lr=lr)
		elif self.args.particles_type=='quaternion':
			if self.args.optimizer=='SGD':
				return optimizers.quaternion_SGD(self.particles.parameters(), lr=lr)

	def get_true_rm(self):
		if self.args.model =='synthetic':
			self.true_particles = self.prior.sample(self.args.N, int(0.1*self.args.num_particles))
			return self.RM_map(self.true_particles)
		else:
			raise NotImplementedError()

	def train(self,save_particles=True):
		print("Starting Training Loop...")
		start_time = time.time()
		best_valid_loss = np.inf
		for iteration in range(self.args.total_iters):
			#scheduler.step()
			loss = self.train_iter(iteration,save_particles=save_particles)
			if not np.isfinite(loss):
				break 
			if self.args.use_scheduler:
				self.scheduler.step(loss)


			if np.mod(iteration,self.args.noise_decay_freq)==0 and iteration>0:
				self.particles.update_noise_level()

		return loss

	def train_iter(self, iteration,save_particles=True):
		self.particles.zero_grad()
		#rm_particles = self.RM_map(self.particles)
		loss = self.loss(self.true_RM)
		loss.backward()
		
		self.optimizer.step()
		
		loss = loss.item()
		save(self.writer,loss,self.particles,iteration,save_particles= save_particles)
		#if np.mod(iteration,100)==0:
		print('Iteration: '+ str(iteration) + ' loss: ' + str(round(loss,3)))
		return loss
def get_kernel(args,dtype, device):

	if args.kernel == 'gaussian':
		return Gaussian(1 , args.kernel_bw, particles_type=args.particles_type, dtype=dtype, device=device)
	else:
		raise NotImplementedError()

def get_particles(args, prior):
	if args.particles_type == 'euclidian':
		return particles.Particles(prior,args.N, args.num_particles ,args.noise_level, args.noise_decay)
	elif args.particles_type== 'quaternion':
		return particles.QuaternionParticles(prior, args.N, args.num_particles ,args.noise_level, args.noise_decay)
	else:
		raise NotImplementedError()

def get_rm_map(args,edges):
	if args.particles_type=='euclidian':
		return particles.RelativeMeasureMap(edges)
	elif args.particles_type=='quaternion':
		return particles.QuaternionRelativeMeasureMap(edges)
	else:
		raise NotImplementedError()

def get_prior(args):
	if args.prior =='mixture_gaussians' and args.particles_type=='euclidian':
		return prior.MixtureGaussianPrior(args.maxNumModes)
	elif args.prior =='gaussian' and args.particles_type=='quaternion':
		return prior.GaussianQuaternionPrior()
	elif args.prior=='bingrham ':
		raise NotImplementedError()
	else:
		raise NotImplementedError()


def get_edges(args):
	if args.model=='synthetic':
		return utils.generate_graph(args.N,args.completeness)
	else:
		raise NotImplementedError()
def get_dtype(args):
	if args.dtype=='float32':
		return torch.float32
	else:
		return torch.float64

def get_scheduler(args, optimizer):
	if args.scheduler == 'ReduceLROnPlateau':
		return torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min',patience = 50,verbose=True, factor = 0.9)

def save(writer,loss,particles,iteration, save_particles=True):

	writer.add_scalars('data/',{"losses":loss},iteration)
	if np.mod(iteration,5)==0:
		print('Saving checkpoint at iteration'+ str(iteration))
		state = {
			'particles': particles,
			'loss': loss,
			'iteration':iteration,
		}
		if not os.path.isdir(writer.logdir +'/checkpoint'):
			os.mkdir(writer.logdir + '/checkpoint')
		torch.save(state,writer.logdir +'/checkpoint/ckpt.iter_'+str(iteration))


