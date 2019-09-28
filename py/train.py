import argparse
import torch
torch.backends.cudnn.benchmark = True


from experiment import experiment


parser = argparse.ArgumentParser(description='PyTorch CIFAR10 Training')
parser.add_argument('--lr', default=.01, type=float, help='learning rate')
parser.add_argument('--resume', '-r', action='store_true', help='resume from checkpoint')
parser.add_argument('--log_dir', default = '/nfs/data/michaela/projects/MMD-flow/code/runs',type= str,  help='log directory for summaries and checkpoints')
parser.add_argument('--log_name', default = 'test',type= str,  help='log directory for summaries and checkpoints')
parser.add_argument('--loss', default = 'mmd',type= str,  help='log directory for summaries and checkpoints')
parser.add_argument('--bandwidth', default=2., type=float, help='learning rate')

parser.add_argument('--total_epochs', default=10000, type=int, help='total number of epochs')
parser.add_argument('--device', default = 0 ,type= int,  help='gpu device')
parser.add_argument('--dtype', default = 'float32' ,type= str,  help='gpu device')
parser.add_argument('--seed', default = 999 ,type= int,  help='gpu device')
parser.add_argument('--batch_size', default = 100 ,type= int,  help='gpu device')
parser.add_argument('--d_int', default = 50 ,type= int,  help='dim data')
parser.add_argument('--d_out', default = 1 ,type= int,  help='dim out feature')
parser.add_argument('--H', default = 3  ,type= int,  help='dim out feature')

parser.add_argument('--center_target', default = 0. ,type= float,  help='gpu device')
parser.add_argument('--sigma_target', default = 1. ,type= float,  help='gpu device')
parser.add_argument('--center_particles', default = 10. ,type= float,  help='gpu device')
parser.add_argument('--sigma_particles', default = .5 ,type= float,  help='gpu device')
parser.add_argument('--num_particles', default = 1000 ,type= int,  help='gpu device')
parser.add_argument('--target_type', default = 'gaussian' ,type= str,  help='gpu device')
parser.add_argument('--particles_type', default = 'gaussian' ,type= str,  help='gpu device')

parser.add_argument('--N_train', default = 1000 ,type= int,  help='gpu device')
parser.add_argument('--N_valid', default = 1000 ,type= int,  help='gpu device')

parser.add_argument('--method', default='noisy', type=str, help='learning rate')

parser.add_argument('--noise_level', default = 1. ,type= float,  help='gpu device')

parser.add_argument('--mean_student', default = 0.001  ,type= float,  help='gpu device')
parser.add_argument('--std_student', default = 1.  ,type= float,  help='gpu device')

parser.add_argument('--teacher', default = 'Spherical' ,type= str,   help='gpu device')
parser.add_argument('--teacher_net', default = 'OneHidden' ,type= str,   help='gpu device')
parser.add_argument('--student_net', default = 'NoisyOneHidden' ,type= str,   help='gpu device')
parser.add_argument('--non_linearity', default = 'quadexp' ,type= str,   help='gpu device')

parser.add_argument('--mean_teacher', default = 0.  ,type= float,  help='gpu device')
parser.add_argument('--std_teacher', default = 1.  ,type= float,  help='gpu device')

parser.add_argument('--num_std', default = 5 ,type= int,  help='gpu device')
parser.add_argument('--num_lr', default = 5 ,type= int,  help='gpu device')
parser.add_argument('--num_noise', default = 1 ,type= int,  help='gpu device')


parser.add_argument('--std_min', default = -3 ,type= float,  help='gpu device')
parser.add_argument('--lr_min', default = -3 ,type= float,  help='gpu device')
parser.add_argument('--noise_min', default = -1 ,type= float,  help='gpu device')

parser.add_argument('--std_max', default = 1 ,type= float,  help='gpu device')
parser.add_argument('--lr_max', default = -1 ,type= float,  help='gpu device')
parser.add_argument('--noise_max', default = 2 ,type= float,  help='gpu device')


parser.add_argument('--mode',  default = 'train' ,type= str,   help='gpu device')
parser.add_argument('--optimizer',  default = 'SGD' ,type= str,   help='gpu device')

parser.add_argument('--log_in_file', action='store_true',  help='gpu device')


parser.add_argument('--noise_decay_freq', default = 1000 ,type= int,  help='gpu device')
parser.add_argument('--noise_decay', default = 0.5 ,type= float,  help='gpu device')
parser.add_argument('--truncate_stds', default = -1. ,type= float,  help='gpu device')


parser.add_argument('--use_scheduler',   action='store_true',  help='gpu device')

parser.add_argument('--bias',   action='store_true',  help='gpu device')


args = parser.parse_args()

exp = experiment(args)



if args.mode=='cv':
	df = exp.cross_validation()
	exp.save_res(df)
elif args.mode=='train':
		exp.train()








