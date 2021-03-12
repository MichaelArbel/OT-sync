
from __future__ import print_function
import argparse
import torch
import numpy as np
#np.random.seed(0)

from sacred import Experiment
from sacred.observers import MongoObserver

from sacred.utils import apply_backspaces_and_linefeeds
from sacred.stflow import LogFileWriter
from collections import namedtuple
import pprint as pp
from copy import deepcopy
import pprint
import yaml
from trainer import Trainer
import pickle


class Struct:
	def __init__(self, **entries):
		self.__dict__.update(entries)

# instantiate a sacred experiment
ex = Experiment("experiment_name")
# add the MongoDB observer
#ex.observers.append(MongoObserver.create())

ex.observers.append(MongoObserver.create(url='mongodb://192.168.213.229:1233',db_name='mongo_db_exp'))

def make_flags(args,config_file):
    if config_file:
        config = yaml.load(open(config_file))
        dic = vars(args)
        all(map(dic.pop, config))
        dic.update(config)
    return args



@ex.config
def my_config():
	"""This is my demo configuration"""
	resume = False
	log_dir= '/nfs/data/michaela/projects/OptSync'
	log_name= 'exp'
	log_in_file=True
	device = 1
	dtype = '64'
	seed = 0
	total_iters = 10000
	lr = 0.01
	use_scheduler = False
	scheduler = 'ReduceLROnPlateau'
	N = 45
	particles_type = 'quaternion'
	num_particles = 100
	prior = 'gaussian'
	kernel_cost = 'power_quaternion'
	kernel_log_bw = 0.
	loss = 'mmd'
	noise_decay_freq = 1000
	noise_decay = 0.5
	noise_level = 1.
	completeness = 0.5
	model = 'synthetic'
	optimizer = 'SGD'
	save = 1
	SH_eps = 0.0001
	SH_max_iter = 1000
	eval_loss = 'sinkhorn'
	freq_eval = 10
	with_weights = 1
	with_noise = False
	product_particles = 0
	true_prior = 'gaussian'
	num_true_particles = 1
	true_product_particles = True
	true_rm_noise_level = -1.
	true_bernoulli_noise = -1.
	unfaithfulness = False
	config_method=''
	config_data = ''
	power = 1.2
	with_backtracking = True
	weights_factor = 0.001
	with_couplings = False
	num_rm_particles = 3
	weight_decay = 0.
	with_edges_splits = False
	batch_size = 10
	lr_step_size = 1000
	lr_decay = .1
	data_path = '../data'
	data_name = 'notredame'
	multi_gpu = False
	weight_penalty = 0.
	decay_lr = 10000

	dicts={ 'log_dir':log_dir,
			'log_name':log_name,
			'log_in_file':log_in_file,
			'lr': lr,
			'resume':resume,
			'dtype':dtype,
			'device':device,
			'optimizer':optimizer,
			'seed':seed,
			'run_id':0,
			'total_iters':total_iters,
			'use_scheduler':use_scheduler,
			'scheduler':scheduler,
			'N':N,
			'particles_type':particles_type,
			'num_particles':num_particles,
			'prior':prior,
			'kernel_cost':kernel_cost,
			'kernel_log_bw':kernel_log_bw,
			'loss':loss,
			'noise_decay_freq':noise_decay_freq,
			'noise_decay':noise_decay,
			'noise_level':noise_level,
			'completeness':completeness,
			'model':model,
			'optimizer':optimizer,
			'save':save,
			'SH_eps':SH_eps,
			'SH_max_iter':SH_max_iter,
			'eval_loss':eval_loss,
			'freq_eval':freq_eval,
			'with_weights':with_weights,
			'with_noise':with_noise,
			'product_particles':product_particles,
			'true_prior': true_prior,
			'num_true_particles': num_true_particles,
			'true_product_particles': true_product_particles,
			'true_rm_noise_level': true_rm_noise_level,
			'true_bernoulli_noise': true_bernoulli_noise,
			'unfaithfulness': unfaithfulness,
			'config_method':config_method,
			'config_data':config_data,
			'power':power,
			'with_backtracking':with_backtracking,
			'weights_factor': weights_factor,
			'with_couplings':with_couplings,
			'num_rm_particles':num_rm_particles,
			'weight_decay':weight_decay,
			'with_edges_splits':with_edges_splits,
			'batch_size':batch_size,
			'lr_step_size':lr_step_size,
			'data_path':data_path,
			'data_name':data_path,
			'lr_decay':lr_decay,
			'multi_gpu':multi_gpu,
			'weight_penalty':weight_penalty,
			'decay_lr':decay_lr,
		}

@ex.automain
@LogFileWriter(ex)
def main(dicts,_run):

	args = Struct(**dicts)
	args = make_flags(args,args.config_method)
	args = make_flags(args,args.config_data)
	args.run_id = _run._id

	pp = pprint.PrettyPrinter(indent=4)
	pp.pprint(vars(args))

	trainer = Trainer(args)
	trainer.train()

	print(' Training Ended ')