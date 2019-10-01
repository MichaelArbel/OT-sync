from __future__ import print_function



import os
import argparse
import time 
import numpy as np
from functools import partial
import copy
import time
import itertools
import sys
import pandas as pd



import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import torch.backends.cudnn as cudnn
import torch.optim as optim
from torch.utils import data
from torch.autograd import Variable
import torchvision
import torchvision.transforms as transforms
from tensorboardX import SummaryWriter



#from model_generator import *
#from Datasets import *
from Utils import * 
from networks import *


class experiment(object):
	def __init__(self,args):
		self.args = args
		self.device = 'cuda:'+str(args.device) if torch.cuda.is_available() and args.device>-1 else 'cpu'


		self.dtype = get_dtype(args)
		self.log_dir = os.path.join(args.log_dir, args.log_name+ '_mode_' + args.mode+'_loss_' + args.loss + '_method_' + args.method )

		if not os.path.isdir(self.log_dir):
			os.mkdir(self.log_dir)
		
		if args.log_in_file:
			self.log_file = open(os.path.join(self.log_dir, 'log.txt'), 'w', buffering=1)
			sys.stdout = self.log_file
			#sys.stderr = self.log_file
		print('Creating writer')
		self.writer = SummaryWriter(self.log_dir)
		#self.multi_exp_writer = SummaryWriter(self.log_dir+'all_exp')
		#print('Loading data')
		#self.data_loaders, self.data_lengths = get_loader(args)
		#self.total_epochs = self.args.total_epochs

		print('==> Building model..')
		# net = VGG('VGG19')
		self.build_model()


	def build_model(self):
		torch.manual_seed(self.args.seed)
		if not self.args.method=='noisy':
			self.args.noise_level = 0.
		self.teacherNet = get_net(self.args,self.dtype,self.device,'teacher')
		self.student = get_net(self.args,self.dtype,self.device,'student')
		self.teacher = get_teacher(self.teacherNet,self.args,self.dtype,self.device)
		self.valid_teacher = get_teacher(self.teacherNet,self.args,self.dtype,self.device)
		
		self.loss = self.get_loss()

		self.optimizer = self.get_optimizer(self.args.lr)
		self.scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(self.optimizer, 'min',patience = 50,verbose=True, factor = 0.9)
		#self.get_reg_dist()

	def get_loss(self):
		if self.args.loss=='mmd':
			return Loss(self.student)
		elif self.args.loss=='diffusion':
			return LossDiffusion(self.student)
		elif self.args.loss=='simple_mmd':
			return SimpleLoss(self.student)
		elif self.args.loss=='sobolev':
			return LossSobolev(self.student)
	def get_optimizer(self,lr):
		if self.args.optimizer=='SGD':
			return optim.SGD(self.student.parameters(), lr=lr)

	def init_student(self,mean,std):
		weights_init_student = partial(weights_init,{'mean':mean,'std':std})
		self.student.apply(weights_init_student)

	def train(self,start_epoch=0,total_iters=0,save_particles=False):
		print("Starting Training Loop...")
		start_time = time.time()
		best_valid_loss = np.inf
		for epoch in range(start_epoch, start_epoch+self.args.total_epochs):
			#scheduler.step()
			total_iters,train_loss = train_epoch(epoch,total_iters,self.loss,self.teacher,self.optimizer,'train',  device=self.device,save_particles=save_particles, writer = self.writer)
			total_iters,valid_loss = train_epoch(epoch, total_iters, self.loss,self.valid_teacher,self.optimizer,'valid',  device=self.device,save_particles=save_particles, writer = self.writer)
			if not np.isfinite(train_loss):
				break 

			if valid_loss < best_valid_loss:
				best_valid_loss = valid_loss
			if self.args.use_scheduler:
				self.scheduler.step(train_loss)
			#scheduler.step(total_loss)


			if np.mod(epoch,self.args.noise_decay_freq)==0 and epoch>0:
				self.loss.student.update_noise_level()
			if np.mod(epoch,10)==0:
				new_time = time.time()
				print("time : " + str(new_time-start_time) )
				start_time = new_time
		return train_loss,valid_loss,best_valid_loss


	def cross_validation(self):
		#all_lr = np.logspace(-3,-1,num=self.args.num_lr)
		#all_std = np.logspace(-3,1,num=self.args.num_std)
		#all_noise = np.logspace(-2,2,num=self.args.num_noise)
		#all_params = [all_lr,all_std,all_noise]
		all_params = set_params(self.args)
		df = pd.DataFrame()
		for i, param in enumerate(all_params):
			lr, std, noise_level = param
			self.args.noise_level = noise_level
			self.args.lr = lr
			self.args.std_student = std
			self.build_model()
			tmp_dir = self.log_dir+'/params_'+str(i)
			if not os.path.isdir(tmp_dir):
				os.mkdir(tmp_dir)
			self.writer = SummaryWriter(tmp_dir)
			if self.args.log_in_file:
				self.log_file = open(os.path.join(tmp_dir, 'log.txt'), 'w', buffering=1)
				sys.stdout = self.log_file
			#self.init_student(self.args.mean_student,std)
			#self.optimizer = self.get_optimizer(lr)
			
			train_loss,val_loss,best_valid_loss = self.train()
			out = {"train_loss":train_loss, "valid_loss":val_loss, "best_valid_loss":best_valid_loss,"lr":lr,"std":std,"noise":noise_level}
			#self.multi_exp_writer.add_hparams({"lr":lr,"std":std,"noise":noise_level},{"train_loss":train_loss, "valid_loss":val_loss, "best_valid_loss":best_valid_loss})
			df = df.append( pd.DataFrame(out, index=[i]),ignore_index=True )
			self.save_res(df)
			#except:
			#print(' Training failed ')
			#self.save_dataframe(out)
		return df

	def save_res(self,df):
		import pickle
		pickle_out = open(self.log_dir+"all_out.pickle","wb")
		pickle.dump(df, pickle_out)
		pickle_out.close()




def set_params(args):
	all_lr = np.logspace(args.lr_min,args.lr_max,num=args.num_lr)
	all_std = np.logspace(args.std_min,args.std_max,num=args.num_std)
	all_noise = np.logspace(args.noise_min,args.noise_max,num=args.num_noise)
	if args.truncate_stds>0:
		all_std = all_std[all_std >=args.truncate_stds]
	all_params = [all_lr,all_std,all_noise]
	return list(itertools.product(*all_params))

def get_teacher(net,args,dtype,device):
	params = {'batch_size': args.batch_size,
		  'shuffle': True,
		  'num_workers': 0}
	if args.teacher=='Spherical':
		teacher  = SphericalTeacher(net,args.N_train,dtype,device)
	return data.DataLoader(teacher, **params)

def get_net(args,dtype,device,net_type):
	non_linearity = get_nonlinearity(args)
	if net_type=='teacher':
		weights_init_net = partial(weights_init,{'mean':args.mean_teacher,'std':args.std_teacher})
		if args.teacher_net=='OneHidden':
			Net = OneHiddenLayer(args.d_int,args.H,args.d_out,non_linearity = non_linearity,bias=args.bias)
	if net_type=='student':
		weights_init_net = partial(weights_init,{'mean':args.mean_student,'std':args.std_student})
		if args.student_net=='NoisyOneHidden':
			Net = NoisyOneHiddenLayer(args.d_int, args.H, args.d_out, args.num_particles, non_linearity = non_linearity, noise_level = args.noise_level,noise_decay=args.noise_decay,bias=args.bias)

	Net.to(device)
	if args.dtype=='float64':
		Net.double()
	
	Net.apply(weights_init_net)
	return Net

def get_nonlinearity(args):
	if args.non_linearity=='quadexp':
		return quadexp()
	elif args.non_linearity=='identity':
		return Identity()

def get_dtype(args):
	if args.dtype=='float32':
		return torch.float32
	else:
		return torch.float64


def weights_init(args,m):
	if isinstance(m, nn.Linear):
		m.weight.data.normal_(mean=args['mean'],std=args['std'])
		if m.bias:
			m.bias.data.normal_(mean=args['mean'],std=args['std'])


def train_epoch(epoch,total_iters,Loss,data_loader, optimizer,phase, device="cuda", save_particles = False, writer = None):

	# Training Loop
	# Lists to keep track of progress



	if phase == 'train':
		Loss.student.train(True)  # Set model to training mode
	else:
		Loss.student.train(False)  # Set model to evaluate mode


	
	cum_loss = 0
	# For each epoch

	# For each batch in the dataloader
	for batch_idx, (inputs, targets) in enumerate(data_loader):
		if phase=="train":
			total_iters += 1
			Loss.student.zero_grad()
			loss = Loss(inputs, targets)
			# Calculate the gradients for this batch
			loss.backward()
		
			# Update G
			#if loss.item()>1e5:
			#	tr.nn.utils.clip_grad_norm_(Loss.student.parameters(), tr.abs(loss)/10000.)
			optimizer.step()
			loss = loss.item()
			cum_loss += loss
			save(writer,loss,Loss.student,total_iters,phase)

		elif phase=='valid':
			loss = Loss(inputs, targets).item()
			cum_loss += loss
	total_loss = cum_loss/(batch_idx+1)
	if phase=='valid':
		save(writer,total_loss,Loss.student,epoch,phase)
	elif phase=='train':
		save(writer,total_loss,Loss.student,epoch,'final_train')
	if np.mod(epoch,10)==0:

		print('Epoch: '+ str(epoch) + ' | ' + phase + ' phase' )
		print( phase + ' loss: ' + str(round(total_loss,3)) )

	return total_iters, total_loss

def save(writer,loss,net,iters,mode, save_particles=True):

	if mode=='train':
		writer.add_scalars('data/'+mode,{"losses":loss},iters)
		if np.mod(iters,5000)==0:
			print('Saving checkpoint at iteration'+ str(iters))
			state = {
				'net': net.state_dict(),
				'loss': loss,
				'iters':iters,
			}
			if not os.path.isdir(writer.logdir +'/checkpoint'):
				os.mkdir(writer.logdir + '/checkpoint')
			tr.save(state,writer.logdir +'/checkpoint/ckpt.iter_'+str(iters))
	elif mode=='valid' or mode=='final_train':
		writer.add_scalars('data/'+mode,{"losses":loss},iters)


