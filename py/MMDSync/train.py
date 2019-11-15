import argparse
import torch
torch.backends.cudnn.benchmark = True
import yaml
from trainer import Trainer

def make_flags(args,config_file):
    if config_file:
        config = yaml.load(open(config_file))
        dic = vars(args)
        all(map(dic.pop, config))
        dic.update(config)
    return args



parser = argparse.ArgumentParser(description='PyTorch CIFAR10 Training')
parser.add_argument('--resume', '-r', action='store_true', help='resume from checkpoint')
parser.add_argument('--log_dir', default = '',type= str,  help='log directory for summaries and checkpoints')
parser.add_argument('--log_name', default = 'exp',type= str,  help='log directory for summaries and checkpoints')
parser.add_argument('--log_in_file', action='store_true',  help='log output in file ')

parser.add_argument('--data_path', default = '../data',type= str,  help='log directory for summaries and checkpoints')
parser.add_argument('--data_name', default = 'notredame',type= str,  help='log directory for summaries and checkpoints')


parser.add_argument('--device', default = 0 ,type= int,  help='gpu device')
parser.add_argument('--dtype', default = 'float64' ,type= str,  help='gpu device')

parser.add_argument('--seed', default = 0 ,type= int,  help='gpu device')

parser.add_argument('--total_iters', default=10000, type=int, help='total number of epochs')
parser.add_argument('--lr', default=.01, type=float, help='learning rate')
parser.add_argument('--use_scheduler',   action='store_true',  help='enables scheduler for learning rate')
parser.add_argument('--scheduler',  default='StepLR',  type=str,  help='enables scheduler for learning rate')
parser.add_argument('--lr_step_size',  default=1000, type=int,  help='enables scheduler for learning rate')
parser.add_argument('--lr_decay',  default=.1, type=float,   help='enables scheduler for learning rate')
parser.add_argument('--decay_lr',  default=10000, type=float,   help='enables scheduler for learning rate')



parser.add_argument('--N', default = 45 ,type= int,  help='num cameras')

parser.add_argument('--particles_type', default = 'quaternion' ,type= str,  help='gpu device')
parser.add_argument('--num_particles', default = 100, type= int,  help='num_particles used in the algorithm')

parser.add_argument('--prior', default='gaussian', type=str, help='sampler for the initial particles')

parser.add_argument('--kernel_cost', default='power_quaternion', type=str, help='kernel type')
parser.add_argument('--kernel_log_bw',default = 0. ,type= float, help='bw of the gaussian kernel')

parser.add_argument('--loss', default = 'mmd',type= str,  help='log directory for summaries and checkpoints')
parser.add_argument('--noise_decay_freq', default = 1000 ,type= int,  help='frequency for decay of the level of noise')
parser.add_argument('--noise_decay', default = 0.5 ,type= float,  help=' decay factor for the level of noise')
parser.add_argument('--noise_level', default = 1. ,type= float,  help='gpu device')
parser.add_argument('--completeness', default = .5 ,type= float,  help='gpu device')


parser.add_argument('--model',  default = 'synthetic' ,type= str,   help='type of model')
parser.add_argument('--optimizer',  default = 'SGD' ,type= str,   help=' scpecify optimizer ')
parser.add_argument('--save',  default = 1 ,type= int,   help=' scpecify optimizer ')
parser.add_argument('--SH_eps',  default = 0.0001 ,type= float,   help=' scpecify optimizer ')
parser.add_argument('--SH_max_iter',  default = 1000 ,type= float,   help=' scpecify optimizer ')
parser.add_argument('--eval_loss',  default = 'sinkhorn' ,type= str,   help=' scpecify optimizer ')
parser.add_argument('--freq_eval',  default = 10 ,type= float,   help=' scpecify optimizer ')
parser.add_argument('--with_weights',  default = 1 ,type= int,   help=' scpecify optimizer ')
parser.add_argument('--with_noise',  action='store_true',  help=' scpecify optimizer ')
parser.add_argument('--product_particles',  default = 0,type= int,   help=' scpecify optimizer ')
parser.add_argument('--run_id',  default = 0,type= int,   help=' scpecify optimizer ')



parser.add_argument('--true_prior',  default='gaussian', type=str,  help=' scpecify optimizer ')
parser.add_argument('--num_true_particles',  default = 1 ,type= int,   help=' scpecify optimizer ')
parser.add_argument('--true_product_particles', default = 1 ,type= int,  help=' scpecify optimizer ')
parser.add_argument('--true_rm_noise_level',  default = -1. ,type= float,   help=' scpecify optimizer ')
parser.add_argument('--true_bernoulli_noise',  default = -1.,type= float,   help=' scpecify optimizer ')

parser.add_argument('--unfaithfulness',  action='store_true',      help=' scpecify optimizer ')
parser.add_argument('--config_method',  default = '',type= str,   help=' scpecify optimizer ')
parser.add_argument('--config_data',  default = '',type= str,   help=' scpecify optimizer ')

parser.add_argument('--power',  default = 1.2,type= float,   help=' scpecify optimizer ')
parser.add_argument('--with_backtracking',  action='store_true',        help=' scpecify optimizer ')


parser.add_argument('--weights_factor',  default = 0.001,type= float,   help=' scpecify optimizer ')
parser.add_argument('--with_couplings',  action='store_true',    help=' scpecify optimizer ')


parser.add_argument('--num_rm_particles',  default = 3,type= int,    help=' scpecify optimizer ')
parser.add_argument('--weight_penalty',  default = 0.,type= int,    help=' weight decay rate ')
parser.add_argument('--with_edges_splits', action='store_true',   help=' scpecify optimizer ')
parser.add_argument('--batch_size',default = 10,type= int,  help=' size of the batch ')
parser.add_argument('--multi_gpu',action='store_true',  help=' use multi-gpu? ')
parser.add_argument('--conjugate',action='store_true',  help=' use multi-gpu? ')
parser.add_argument('--GT_mode',action='store_true',  help=' use multi-gpu? ')
parser.add_argument('--err_tol',  default = 0.0001,type= float,   help=' scpecify optimizer ')



args = parser.parse_args()
args = make_flags(args,args.config_method)
args = make_flags(args,args.config_data)

trainer = Trainer(args)
trainer.train()
