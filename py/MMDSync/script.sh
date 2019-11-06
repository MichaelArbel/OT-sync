
###########################################################################################
##  To run this script either install sacred, or replace each command in the following way:
###########################################################################################

#python train_sacred.py with device=1 loss=sinkhorn log_name=toy_exp lr=.1 freq_eval=10 total_iters=10000 num_particles=10 completeness=0.1 kernel_cost=power_quaternion with_weights=0  product_particles=0 & 

# becomes:

#python train.py --device=1 --loss=sinkhorn --log_name=toy_exp --lr=.1 --freq_eval=10 --total_iters=10000 --num_particles=10 --completeness=0.1 --kernel_cost=power_quaternion --with_weights=0 --product_particles=0 & 





#######  I-  Toy example   N=45, truemodes=1,   num_particles=10, completeness=0.1

###### 1- Power Quaternion

## 1.1- sinkhorn ###########

#python train_sacred.py with device=1 loss=sinkhorn log_name=toy_exp lr=.1 freq_eval=10 total_iters=10000 num_particles=10 completeness=0.1 kernel_cost=power_quaternion with_weights=0  product_particles=0 & 

#python train_sacred.py with device=0 loss=sinkhorn log_name=toy_exp lr=.1 freq_eval=10 total_iters=10000 num_particles=10 completeness=0.1 kernel_cost=power_quaternion with_weights=1  product_particles=0 & 

#python train_sacred.py with device=2 loss=sinkhorn log_name=toy_exp lr=.1 freq_eval=10 total_iters=10000 num_particles=10 completeness=0.1 kernel_cost=power_quaternion with_weights=0  product_particles=1 & 

# python train_sacred.py with device=0 loss=sinkhorn log_name=toy_exp lr=.1 freq_eval=10 total_iters=10000 num_particles=10 completeness=0.1 kernel_cost=power_quaternion with_weights=1  product_particles=1 & 


## 1.2- MMD ###############

# python train_sacred.py with device=1 loss=mmd log_name=toy_exp lr=.1 freq_eval=10 total_iters=10000 num_particles=10 completeness=0.1 kernel_cost=power_quaternion with_weights=0  product_particles=0 with_noise=1 noise_level=0.5 noise_decay=0.5 & 

# python train_sacred.py with device=2 loss=mmd log_name=toy_exp lr=.1 freq_eval=10 total_iters=10000 num_particles=10 completeness=0.1 kernel_cost=power_quaternion with_weights=1  product_particles=0 with_noise=1 noise_level=0.5 noise_decay=0.5  &

# python train_sacred.py with device=0 loss=mmd log_name=toy_exp lr=.1 freq_eval=10 total_iters=10000 num_particles=10 completeness=0.1 kernel_cost=power_quaternion with_weights=0  product_particles=1 with_noise=1 noise_level=0.5 noise_decay=0.5 & 

# python train_sacred.py with device=1 loss=mmd log_name=toy_exp lr=.1 freq_eval=10 total_iters=10000 num_particles=10 completeness=0.1 kernel_cost=power_quaternion with_weights=1  product_particles=1 with_noise=1 noise_level=0.5 noise_decay=0.5 & 


###### 2- Squared Euclidean 

## 2.1- sinkhorn ###########
# python train_sacred.py with device=2 loss=sinkhorn log_name=toy_exp lr=.1 freq_eval=10 total_iters=10000 num_particles=10 completeness=0.1 kernel_cost=squared_euclidean with_weights=0  product_particles=0 & 

# python train_sacred.py with device=0 loss=sinkhorn log_name=toy_exp lr=.1 freq_eval=10 total_iters=10000 num_particles=10 completeness=0.1 kernel_cost=squared_euclidean with_weights=1  product_particles=0 & 

# python train_sacred.py with device=1 loss=sinkhorn log_name=toy_exp lr=.1 freq_eval=10 total_iters=10000 num_particles=10 completeness=0.1 kernel_cost=squared_euclidean with_weights=0  product_particles=1 & 

# python train_sacred.py with device=2 loss=sinkhorn log_name=toy_exp lr=.1 freq_eval=10 total_iters=10000 num_particles=10 completeness=0.1 kernel_cost=squared_euclidean with_weights=1  product_particles=1 & 


## 2.2- mmds ###########

# python train_sacred.py with device=0 loss=mmd log_name=toy_exp lr=.1 freq_eval=10 total_iters=10000 num_particles=10 completeness=0.1 kernel_cost=squared_euclidean with_weights=0  product_particles=0 with_noise=1 noise_level=0.5 noise_decay=0.5 & 

# python train_sacred.py with device=1 loss=mmd log_name=toy_exp lr=.1 freq_eval=10 total_iters=10000 num_particles=10 completeness=0.1 kernel_cost=squared_euclidean with_weights=1  product_particles=0 with_noise=1 noise_level=0.5 noise_decay=0.5 & 

# python train_sacred.py with device=2 loss=mmd log_name=toy_exp lr=.1 freq_eval=10 total_iters=10000 num_particles=10 completeness=0.1 kernel_cost=squared_euclidean with_weights=0  product_particles=1 with_noise=1 noise_level=0.5 noise_decay=0.5 & 

# python train_sacred.py with device=0 loss=mmd log_name=toy_exp lr=.1 freq_eval=10 total_iters=10000 num_particles=10 completeness=0.1 kernel_cost=squared_euclidean with_weights=1  product_particles=1 with_noise=1 noise_level=0.5 noise_decay=0.5 & 



####################################################################
####################################################################

python -m ipdb train.py --device=2 --log_name=toy_exp --config_method=configs/sinkhorn.yaml  --model='real_data' --data_path='../data/' --data_name='notredame' --lr=.1  &



python -m ipdb train.py --device=-2 --log_name=toy_exp --config_method=configs/sinkhorn.yaml  --model='real_data' --data_path='../data/' --data_name='notredame' --lr=.1  &




