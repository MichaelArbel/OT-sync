import os
import time 
import numpy as np
import sys
import torch
import torch.optim as optim
import torch.nn as nn

import utils
import pprint
import mmd
import sinkhorn
import particles
import prior
import optimizers
from kernel.gaussian import Gaussian,ExpQuaternionGeodesicDist,ExpPowerQuaternionGeodesicDist
import pickle
import data_loader as dl

torch.backends.cudnn.benchmark=True
torch.manual_seed(0)
torch.backends.cudnn.deterministic = True

class Trainer(object):
	def __init__(self,args):
		torch.manual_seed(args.seed)
		self.args = args
		if args.device >-1:
			self.device = 'cuda:'+str(args.device) if torch.cuda.is_available() and args.device>-1 else 'cpu'
		elif args.device==-1:
			self.device = 'cuda'
		elif args.device==-2:
			self.device = 'cpu'
		self.dtype = get_dtype(args)
		
		self.log_dir = make_log_dir(args)

		if args.log_in_file:
			self.log_file = open(os.path.join(self.log_dir, 'log.txt'), 'w', buffering=1)
			sys.stdout = self.log_file

		pp = pprint.PrettyPrinter(indent=4)
		pp.pprint(vars(args))

		#print('Creating writer')
		#self.writer = SummaryWriter(self.log_dir)

		print('==> Building model..')
		self.build_model()


	def build_model(self):
		torch.manual_seed(self.args.seed)
		np.random.seed(self.args.seed)
		self.get_gt_data()

		
		self.RM_map = get_rm_map(self.args,self.edges)
		self.prior = get_prior(self.args, self.dtype, self.device)
		
		self.particles = get_particles(self.args, self.prior, self.edges.shape[0] )
		
		self.loss_class = self.get_loss()
		self.eval_loss = self.get_eval_loss()
		self.eval_loss_abs = self.get_eval_loss_abs()
		if torch.cuda.device_count()   > 1 and self.args.multi_gpu:
			self.loss = nn.DataParallel(self.loss_class)
			self.eval_loss = nn.DataParallel(self.eval_loss )
			print("Let's use", torch.cuda.device_count(), "GPUs!")
		else:
			self.loss = self.loss_class
			
		self.loss.to(self.device)
		self.eval_loss.to(self.device)

		self.optimizer = self.get_optimizer(self.args.lr)
		self.scheduler = get_scheduler(self.args, self.optimizer)

		self.old_loss = np.inf
		if self.args.with_edges_splits:
			self.sub_indices,self.sub_edges =  self.split_edges()
	def get_gt_data(self):
		if self.args.model=='synthetic':
			self.edges, self.G = get_edges(self.args)
			self.edges = np.dtype('int64').type(self.edges)
			self.true_RM, self.true_RM_weights = self.get_true_rm()
			self.eval_idx =None
		elif self.args.model=='real_data' and self.args.data_name=='notredame':
			self.edges, self.G, self.true_RM, self.true_RM_weights, self.true_particles , self.true_weights,self.eval_idx = dl.data_loader_notredame(self.args.data_path, self.args.data_name, self.dtype,self.device)
			self.args.N = len(self.true_particles)
		elif self.args.model=='real_data' and self.args.data_name=='artsquad':
			self.edges, self.G, self.true_RM, self.true_RM_weights, self.true_particles , self.true_weights,self.eval_idx = dl.data_loader_artsquad(self.args.data_path, self.args.data_name, self.dtype,self.device)
			self.args.N = len(self.true_particles)
		
		elif self.args.model=='real_data' and self.args.data_name=='marker':
			self.edges, self.G, self.true_RM, self.true_RM_weights, self.true_particles , self.true_weights,self.eval_idx = dl.data_loader_multiparticles(self.args.data_path, self.args.data_name, self.dtype,self.device)
			self.args.N = len(self.true_particles)
		else:
			self.edges, self.G, self.true_RM, self.true_RM_weights, self.true_particles, self.true_weights, self.eval_idx = dl.data_loader_notredame(
			self.args.data_path, self.args.data_name, self.dtype, self.device)
			self.args.N = len(self.true_particles)
			#self.true_weights = (1./true_args.num_particles)*torch.ones([true_args.N, true_args.num_particles], dtype=self.true_particles.dtype, device = self.true_particles.device )
			
	def split_edges(self):
		
		num_splits = int(self.edges.shape[0]/self.args.batch_size)
		if not np.mod(self.edges.shape[0],self.args.batch_size) == 0:
			num_splits +=1
		edges_indices = np.array(range(self.edges.shape[0]))
		sub_edges = np.array_split(self.edges,num_splits)
		split_indices = np.array_split(edges_indices,num_splits)
		return split_indices,sub_edges

	def get_loss(self):
		kernel = get_kernel(self.args,self.dtype, self.device)
		if self.args.loss=='mmd':
			return mmd.MMD_weighted(kernel, self.particles,self.RM_map, with_noise = self.args.with_noise)
		elif self.args.loss=='sinkhorn':
			return sinkhorn.Sinkhorn_weighted(kernel, self.particles,self.RM_map,self.args.SH_eps)
		else:
			raise NotImplementedError()
	def get_optimizer(self,lr):
		if self.args.particles_type=='euclidian':

			if self.args.optimizer=='SGD':
				return optim.SGD(self.particles.parameters(), lr=lr)
		elif self.args.particles_type=='quaternion':
			if self.args.optimizer=='SGD':
				return optimizers.quaternion_SGD(self.particles.parameters(), lr=lr, weights_factor= self.args.weights_factor)
			elif  self.args.optimizer=='SGD_unconstrained':
				return optimizers.quaternion_SGD_unconstrained(self.particles.parameters(), lr=lr, weights_factor= self.args.weights_factor,weight_penalty=self.args.weight_penalty)

	def get_true_rm(self):
		if self.args.model =='synthetic':
			true_args = make_true_dict(self.args)

			true_prior = get_prior(true_args, self.dtype, self.device)
			true_RM_map = get_true_rm_map(true_args, self.edges, self.dtype, self.device)
			#num_particles = self.args.num_true_particles 
			self.true_particles = true_prior.sample(true_args.N, true_args.num_particles)
			# Fixing the pose of the first camera!
			self.true_particles[0,:,0] = 1.
			self.true_particles[0,:,1:] = 0.
			# Weights are uniform
			self.true_weights = (1./true_args.num_particles)*torch.ones([true_args.N, true_args.num_particles], dtype=self.true_particles.dtype, device = self.true_particles.device )
			
			rm, rm_weights = true_RM_map(self.true_particles,  self.true_weights ,self.edges)
				
			return rm, rm_weights
		else:
			raise NotImplementedError()
	def get_eval_loss(self):
		if self.args.eval_loss=='sinkhorn' or self.args.eval_loss=='kbest':
			return sinkhorn.SinkhornEval(self.particles, self.RM_map, self.args.SH_eps, self.args.SH_max_iter,'quaternion')

	def get_eval_loss_abs(self):
		if self.args.eval_loss=='sinkhorn':
			return sinkhorn.SinkhornEvalAbs(self.particles, self.args.SH_eps, self.args.SH_max_iter,'quaternion',self.eval_idx)
		elif self.args.eval_loss=='kbest':
			return sinkhorn.SinkhornEvalKBestAbs(self.particles, self.args.SH_eps, self.args.SH_max_iter, 'quaternion', self.eval_idx)
		#elif self.args.eval_loss == 'mmd':
		#	kernel = get_kernel(self.args, self.dtype, self.device)
		#	return mmd.MMD_weighted(kernel, self.particles, self.RM_map, with_noise=self.args.with_noise)
		#	return sinkhorn.SinkhornEvalAbs(self.particles, self.args.SH_eps, self.args.SH_max_iter, 'quaternion',self.eval_idx)

	def train(self):
		print("Starting Training Loop...")
		start_time = time.time()
		best_valid_loss = np.inf
		with_config = True
		#self.initialize()
		for iteration in range(self.args.total_iters):
			#scheduler.step()
			loss = self.train_iter(iteration)
			#if loss < 0.158:
			#	print(loss)
			if not np.isfinite(loss):
				break 
			if self.args.use_scheduler:
				#self.scheduler.step(loss)
				self.scheduler.step()
			if np.mod(iteration,self.args.noise_decay_freq)==0 and iteration>0:
				self.particles.update_noise_level()
			if np.mod(iteration, self.args.freq_eval)==0:
				with torch.no_grad():
					out = self.eval(iteration, loss,with_config=with_config)
				with_config=False
				if self.args.save==1:
					save_pickle(out, os.path.join(self.log_dir, 'data'), name =  'iter_'+ str(iteration).zfill(8))
			#save(self.writer,loss_val,self.particles,iteration,save_particles= save_particles)

		return loss

	def initialize(self):
		print('initiallizing particles')
		start = time.time()
		num_iters = 10
		K = self.edges.shape[0]
		M = self.true_RM.shape[1]
		N = self.particles.data.shape[0]
		d = self.particles.data.shape[2]
		cpu_true_RM = self.true_RM
		init_particles = torch.zeros([N,M,d], dtype=self.true_RM.dtype, device = cpu_true_RM.device)
		
		
		I = [tuple(l) for l in self.edges]
		#I = list(self.G.edges())
		N = len(self.G.nodes())

		init_particles[0,:,0] = 1.
		j = 0
		done_set = set([])
		cur_set = set([0])
		done  = False

		while not done:
			cur_node = cur_set.pop()			
			successors = set([n for n in self.G.successors(cur_node)])
			predecessors = set([n for n in self.G.predecessors(cur_node)])
			predecessors = predecessors.difference(done_set)
			predecessors = predecessors.difference(cur_set)
			successors = successors.difference(done_set)
			successors = successors.difference(cur_set)
			suc = list(successors)
			K = [I.index((cur_node,s)) for s in suc]
			cp = 1.*cpu_true_RM[K,:,:]
			cp[:,:,1:]*=-1
			if len(K)>0:
				init_particles[suc,:,:] = utils.quaternion_prod(cp, init_particles[cur_node,:,:].unsqueeze(0).repeat(len(K),1,1))
			pre = list(predecessors)
			K = [I.index((p,cur_node)) for p in pre]
			if len(K)>0:
				init_particles[pre,:,:] = utils.quaternion_prod(cpu_true_RM[K,:,:],init_particles[cur_node,:,:].unsqueeze(0).repeat(len(K),1,1))
			mask = init_particles[:,:,0]<0
			init_particles[mask]*=-1
			cur_set.update(successors)
			cur_set.update(predecessors)

			done_set.update([cur_node])
			
			#print('curset: '+str(len(cur_set)) +  ', done_set: '+str(len(done_set)) )
			done = (len(done_set)==N)
		N,num_particles, _ = self.particles.data.shape
		mask_int =torch.multinomial(self.true_RM_weights[0,:],num_particles, replacement=True)
		init_particles = init_particles.to(self.true_RM.device)
		self.particles.data.data = init_particles[:,mask_int,:]
		#self.particles.data.data = self.true_particles[:,mask_int,:]

		#self.particles.data.data = 1.* self.true_particles
		
		#self.particles.data.data = particles.add_noise_quaternion(self.prior,self.particles.data, 0.001)
		#mask = self.particles.data<0
		#self.particles.data.data[mask]*=-1.
		end = time.time()
		print('assigned value in '+str(end-start) + 's')




	def train_iter(self, iteration):
		start = time.time()
		self.particles.zero_grad()

		loss = self.mini_batch_iter(with_backward = True)
		#self.backtracking(loss)
		self.update_gradient(loss,iteration)
		#loss = self.mini_batch_iter(with_backward=True)
		if loss> self.old_loss:
			print('increasing loss')
		#	self.optimizer.param_groups[0]['lr'] =0.5*self.optimizer.param_groups[0]['lr']
		#	print('decreased lr')
		self.old_loss = loss
		self.optimizer.param_groups[0]['lr'] = self.args.lr/(1 + np.sqrt(iteration/self.args.decay_lr))
		#if iteration==500:
		#	self.optimizer.param_groups[0]['lr']*=0.1

		if iteration==1264:
			print('bug here')
		end = time.time()

		print('Iteration: '+ str(iteration) + ' loss: ' + str(round(loss,3))  + ' lr ' + str(self.optimizer.param_groups[0]['lr']) + ' in: ' + str(end-start) +'s' )


		return loss
	def update_gradient(self,loss,iteration):
		if self.args.with_backtracking:
			self.backtracking(loss)
		else:
			if self.args.particles_type=='euclidian':
				self.optimizer.step()
			else:
				self.optimizer.step(loss=loss)


	def mini_batch_iter(self,with_backward):
		if self.args.with_edges_splits: 
			total_loss = 0.
			count = 0
			for k, edges in zip(self.sub_indices,self.sub_edges):
				#self.loss_class.rm_map.edges = edges
				edges = torch.from_numpy(edges)
				loss = self.loss(self.true_RM[k,:], self.true_RM_weights[k,:],edges)
				loss = torch.sum(loss)
				if with_backward:
					loss.backward()
					#self.update_gradient(loss.item(),count)
				total_loss+=loss.item()
				count += 1 
		else:
			edges = torch.from_numpy(self.edges)
			total_loss =self.loss(self.true_RM, self.true_RM_weights,edges)
			if with_backward:
				total_loss.backward()
				#self.update_gradient(total_loss.item(),0)
			total_loss = total_loss.item()
		return total_loss
	def mini_batch_iter_eval_loss(self):
		if self.args.with_edges_splits: 
			total_loss = 0.
			for k, edges in zip(self.sub_indices,self.sub_edges):
				#self.loss_class.rm_map.edges = edges
				edges = torch.from_numpy(edges)
				#rm, rm_weights =self.RM_map(self.particles.data, self.particles.weights(),edges )
				loss = self.eval_loss(self.true_RM[k,:],self.true_RM_weights[k,:],edges)
				loss = torch.sum(loss)
				total_loss+=loss.item()
		else:
			edges = torch.from_numpy(self.edges)
			#rm, rm_weights =self.RM_map(self.particles.data, self.particles.weights(),edges )
			total_loss = self.eval_loss(self.true_RM,self.true_RM_weights,edges)
			total_loss = total_loss.item()		

		return total_loss/len(self.edges)
	def backtracking(self,loss):
		self.optimizer.keep_weights()
		done = False
		count = 0
		count_max = 1
		while not done:
			self.optimizer.reset_weights()
			self.optimizer.step(loss=loss)
			#self.optimizer.step(loss=None)
			with torch.no_grad():
				new_loss = self.mini_batch_iter(with_backward=False)
				#new_loss = self.loss(self.true_RM, self.true_RM_weights).item()
			done = new_loss<=loss or count>count_max
			count +=1
			self.optimizer.decrease_lr()
		self.optimizer.reset_lr(self.args.lr)


	def backtracking_2(self,loss):
		self.optimizer.keep_weights()
		done = False
		count = 0
		count_max = 1
		#self.optimizer.reset_weights()
		self.optimizer.step(loss=loss)
			#self.optimizer.step(loss=None)
		with torch.no_grad():
			new_loss = self.mini_batch_iter(with_backward=False)
				#new_loss = self.loss(self.true_RM, self.true_RM_weights).item()
		if new_loss>loss:
			self.optimizer.keep_weights()
			self.particles.data.grad*=-1.
			self.optimizer.step(loss=loss)
			#self.optimizer.decrease_lr()
		#self.optimizer.reset_lr(self.args.lr)

	def eval(self,iteration, loss_val, with_config=False):
		out ={}
		if with_config:
			out = {"type": self.args.particles_type, "N":self.args.N , "numParticles":self.args.num_particles, "numEdges":len(self.edges), "completeness":self.args.completeness,"I":self.edges  }
			out['true_RM'] = self.true_RM.cpu().detach().numpy()
			out['true_particles'] = self.true_particles.cpu().detach().numpy()
			out['true_weights'] = self.true_weights.cpu().detach().numpy()

		out['loss'] = loss_val
		out['particles'] = self.particles.data.cpu().detach().numpy()
		out['time'] = time.time()
		out['iteration'] = iteration
		if self.args.with_couplings:
			w, c = self.particles.weights()
			out['couplings'] = c.cpu().detach().numpy()
		else:
			w = self.particles.weights()
		out['eval_RM_dist'] =   self.mini_batch_iter_eval_loss()
		out['weights'] = w.cpu().detach().numpy()
		#if self.args.model =='synthetic':
		out['eval_dist'] =   self.eval_loss_abs(self.true_particles, self.true_weights).item()
		out['avg_min_dist'],out['median_min_dist'] = self.best_dist() 
	
			
		print('Sinkhorn distance 	absolute poses '+ str(out['eval_dist']))
		print('Sinkhorn distance 	relative poses '+ str(out['eval_RM_dist']))
		print('Min distance 		absolute poses '+ str(out['avg_min_dist']))
		print('Median distance 		absolute poses '+ str(out['median_min_dist']))

		return out
	def best_dist(self):
		dist = utils.quaternion_geodesic_distance(self.true_particles,self.particles.data)
		min_dist,_ = torch.min(dist,dim=-1)
		avg_best_dist = torch.mean(min_dist,dim=-1)

		median_best_dist = np.median(avg_best_dist[1:].detach().cpu().numpy())
		avg_best_dist = torch.mean(avg_best_dist[1:]).item()
		return avg_best_dist,median_best_dist

def get_kernel(args,dtype, device):

	if args.kernel_cost == 'squared_euclidean':
		return Gaussian(1 , args.kernel_log_bw, particles_type=args.particles_type, dtype=dtype, device=device)
	elif args.kernel_cost == 'quaternion':
		return ExpQuaternionGeodesicDist(1 , args.kernel_log_bw, particles_type=args.particles_type, dtype=dtype, device=device)
	elif args.kernel_cost == 'power_quaternion' or args.kernel_cost == 'sum_power_quaternion':
		return ExpPowerQuaternionGeodesicDist(args.power,1 , args.kernel_log_bw, particles_type=args.particles_type, dtype=dtype, device=device)
	elif args.kernel_cost == 'sinkhorn_gaussian':
		return None
	else:
		raise NotImplementedError()

def get_particles(args, prior,num_edges):

	if args.particles_type == 'euclidian':
		return particles.Particles(prior,args.N, args.num_particles, num_edges,args.with_weights, args.with_couplings, args.product_particles, args.noise_level, args.noise_decay)
	elif args.particles_type== 'quaternion':
		return particles.QuaternionParticles(prior, args.N, args.num_particles ,num_edges, args.with_weights, args.with_couplings, args.product_particles,args.noise_level, args.noise_decay)
	else:
		raise NotImplementedError()

def get_rm_map(args,edges):
	if args.kernel_cost=='quaternion' or args.kernel_cost=='power_quaternion':
		grad_type='quaternion'
	else:
		grad_type='euclidean'
	if args.particles_type=='euclidian':
		#if args.with_weights==1:
		return particles.RelativeMeasureMapWeights(edges,grad_type)
		#else:
		#	return particles.RelativeMeasureMap(edges,grad_type)
	elif args.particles_type=='quaternion':
		#if args.with_weights==1:
		if args.product_particles==1 and args.with_couplings:
			return particles.QuaternionRelativeMeasureMapWeightsCouplings(edges,grad_type)
		elif args.product_particles==1:
			return particles.QuaternionRelativeMeasureMapWeightsProduct(edges,grad_type)
		else:
			return particles.QuaternionRelativeMeasureMapWeights(edges,grad_type)
		#else:
		#	if args.product_particles==1:
		#		return particles.QuaternionRelativeMeasureMapProduct(edges,grad_type)
		#	else:
		#		return particles.QuaternionRelativeMeasureMap(edges,grad_type)
	else:
		raise NotImplementedError()


def get_true_rm_map(args,edges, dtype, device):
	if args.kernel_cost=='quaternion' or args.kernel_cost=='power_quaternion':
		grad_type='quaternion'
	else:
		grad_type='euclidean'
	if args.true_rm_noise_level>0.:
		noise_sampler = get_prior(args, dtype, device)
	else:
		noise_sampler= None
	if args.particles_type=='euclidian':
		return particles.RelativeMeasureMapWeights(edges,grad_type)
	elif args.particles_type=='quaternion':
		if args.product_particles==1:
			return particles.QuaternionRelativeMeasureMapWeightsProductPrior(edges,args.num_rm_particles, grad_type, noise_sampler,args.true_rm_noise_level,args.true_bernoulli_noise,args.unfaithfulness)
		else:
			return particles.QuaternionRelativeMeasureMapWeights(edges,grad_type, noise_sampler,args.true_rm_noise_level,args.true_bernoulli_noise,args.unfaithfulness)
	else:
		raise NotImplementedError()

def get_prior(args, dtype, device):
	if args.prior =='mixture_gaussians' and args.particles_type=='euclidian':
		return prior.MixtureGaussianPrior(args.maxNumModes, dtype, device)
	elif args.prior =='gaussian' and args.particles_type=='quaternion':
		return prior.GaussianQuaternionPrior(dtype, device)
	elif args.prior == 'gaussian' and args.particles_type=='euclidian':
		return prior.GaussianPrior(dtype,device)
	elif args.prior=='bingham ':
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
		return torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min',patience = args.lr_step_size,verbose=True, factor = args.lr_decay)
	elif args.scheduler == 'StepLR':
		return torch.optim.lr_scheduler.StepLR(optimizer, step_size = args.lr_step_size,gamma = args.lr_decay)

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

def save_pickle(out,exp_dir,name):
	os.makedirs(exp_dir, exist_ok=True)
	with  open(os.path.join(exp_dir,name+".pickle"),"wb") as pickle_out:
		pickle.dump(out, pickle_out)



class Struct:
	def __init__(self, **entries):
		self.__dict__.update(entries)

def make_true_dict(args):
	true_args= {}
	true_args['prior'] = args.true_prior
	true_args['particles_type']= args.particles_type
	true_args['N'] = args.N
	true_args['num_particles'] = args.num_true_particles
	true_args['product_particles'] = args.true_product_particles
	true_args['kernel_cost']	 = args.kernel_cost
	true_args['true_rm_noise_level'] = args.true_rm_noise_level
	true_args['true_bernoulli_noise'] = args.true_bernoulli_noise
	true_args['unfaithfulness'] = args.unfaithfulness
	true_args['num_rm_particles'] = args.num_rm_particles
	true_args = Struct(**true_args)
	return true_args

def make_log_dir(args):
	if args.config_method:
		config_name = args.config_method.split('configs/')[1].split('.yaml')[0]
		log_dir = os.path.join(args.log_dir,args.log_name, config_name )
	else:
		if args.product_particles==1:
			rmp_map = '_RM_product'
		else:
			rmp_map = '_RM_joint'
		if args.with_weights:
			weights = '_with_weights'
		else:
			weights = '_no_weights'
		if args.unfaithfulness:
			unfaithfulness = 'true'
		else:
			unfaithfulness = 'false'
		if args.true_product_particles:
			true_prod = 'prod_true'
		else:
			true_prod = 'prod_false'

		model_name = args.model  + '_' +  true_prod + '_N_' +str(args.num_true_particles) + '_noise_' +str(args.true_rm_noise_level)+'_B_noise_' + str(args.true_bernoulli_noise) + '_unfaithfulness_' + str(unfaithfulness)
		method_name =  'Np_' +str(args.num_particles) + '_opt_'+args.optimizer+ '_pow_' + str(args.power) + '_loss_' + args.loss + '_kernel_' + args.kernel_cost+ weights + rmp_map
		log_dir = os.path.join(args.log_dir, args.log_name,  model_name , method_name, str(args.run_id))

	if not os.path.isdir(log_dir):
		from pathlib import Path
		path   = Path(log_dir)
		path.mkdir(parents=True, exist_ok=True)
	return log_dir
		#os.mkdir(self.log_dir)
		

