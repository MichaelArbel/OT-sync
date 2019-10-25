import argparse
import torch
torch.backends.cudnn.benchmark = True

from trainer import Trainer


parser = argparse.ArgumentParser(description='PyTorch CIFAR10 Training')
parser.add_argument('--resume', '-r', action='store_true', help='resume from checkpoint')
parser.add_argument('--log_dir', default = '',type= str,  help='log directory for summaries and checkpoints')
parser.add_argument('--log_name', default = 'exp',type= str,  help='log directory for summaries and checkpoints')
parser.add_argument('--log_in_file', action='store_true',  help='log output in file ')


parser.add_argument('--device', default = 0 ,type= int,  help='gpu device')
parser.add_argument('--dtype', default = 'float32' ,type= str,  help='gpu device')
parser.add_argument('--seed', default = 999 ,type= int,  help='gpu device')

parser.add_argument('--total_iters', default=10000, type=int, help='total number of epochs')
parser.add_argument('--lr', default=.01, type=float, help='learning rate')
parser.add_argument('--use_scheduler',   action='store_true',  help='enables scheduler for learning rate')
parser.add_argument('--scheduler',  default='ReduceLROnPlateau',  type=str,  help='enables scheduler for learning rate')



parser.add_argument('--N', default = 45 ,type= int,  help='num cameras')

parser.add_argument('--particles_type', default = 'quaternion' ,type= str,  help='gpu device')
parser.add_argument('--num_particles', default = 100, type= int,  help='num_particles used in the algorithm')

parser.add_argument('--prior', default='gaussian', type=str, help='sampler for the initial particles')

parser.add_argument('--kernel', default='laplacequaternion', type=str, help='kernel type')
parser.add_argument('--kernel_log_bw',default = 0. ,type= float, help='bw of the gaussian kernel')

parser.add_argument('--loss', default = 'mmd',type= str,  help='log directory for summaries and checkpoints')
parser.add_argument('--noise_decay_freq', default = 1000 ,type= int,  help='frequency for decay of the level of noise')
parser.add_argument('--noise_decay', default = 0.5 ,type= float,  help=' decay factor for the level of noise')
parser.add_argument('--noise_level', default = 1. ,type= float,  help='gpu device')
parser.add_argument('--completeness', default = .5 ,type= float,  help='gpu device')


parser.add_argument('--model',  default = 'synthetic' ,type= str,   help='type of model')
parser.add_argument('--optimizer',  default = 'SGD' ,type= str,   help=' scpecify optimizer ')
parser.add_argument('--save',  default = 1 ,type= int,   help=' scpecify optimizer ')
parser.add_argument('--SH_eps',  default = 0.001 ,type= float,   help=' scpecify optimizer ')
parser.add_argument('--SH_max_iter',  default = 100 ,type= float,   help=' scpecify optimizer ')
parser.add_argument('--eval_loss',  default = 'sinkhorn' ,type= str,   help=' scpecify optimizer ')
parser.add_argument('--freq_eval',  default = 10 ,type= float,   help=' scpecify optimizer ')
parser.add_argument('--with_weights',  default = 1 ,type= int,   help=' scpecify optimizer ')
parser.add_argument('--with_noise',  default = 0,type= int,   help=' scpecify optimizer ')
parser.add_argument('--product_particles',  default = 0,type= int,   help=' scpecify optimizer ')
parser.add_argument('--run_id',  default = 0,type= int,   help=' scpecify optimizer ')





args = parser.parse_args()

trainer = Trainer(args)
trainer.train()
