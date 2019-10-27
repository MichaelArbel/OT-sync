# # no noise

# python train_sacred.py with device=0 loss=mmd log_name=new_exp lr=.01 freq_eval=10 num_particles=10 with_weights=0 completeness=0.1 with_noise=0 kernel=gaussian &
# python train_sacred.py with device=1 loss=mmd log_name=new_exp lr=.01 freq_eval=10 num_particles=10 with_weights=0 completeness=0.1 with_noise=0 kernel=laplacequaternion &
# python train_sacred.py with device=0 loss=mmd log_name=new_exp lr=.01 freq_eval=10 num_particles=10 with_weights=0 completeness=0.1 with_noise=0 kernel=gaussianquaternion &



# # # with noise

# python train_sacred.py with device=1 loss=mmd log_name=new_exp lr=.01 freq_eval=10 num_particles=10 with_weights=0 completeness=0.1 with_noise=1 kernel=gaussian &
# python train_sacred.py with device=0 loss=mmd log_name=new_exp lr=.01 freq_eval=10 num_particles=10 with_weights=0 completeness=0.1 with_noise=1 kernel=laplacequaternion & 
# python train_sacred.py with device=1 loss=mmd log_name=new_exp lr=.01 freq_eval=10 num_particles=10 with_weights=0 completeness=0.1 with_noise=1 kernel=gaussianquaternion &



# # # no noise

# python train_sacred.py with device=0 loss=mmd log_name=new_exp lr=.01 freq_eval=10 num_particles=10 with_weights=1 completeness=0.1 with_noise=0 kernel=gaussian
# python train_sacred.py with device=1 loss=mmd log_name=new_exp lr=.01 freq_eval=10 num_particles=10 with_weights=1 completeness=0.1 with_noise=0 kernel=laplacequaternion
# python train_sacred.py with device=0 loss=mmd log_name=new_exp lr=.01 freq_eval=10 num_particles=10 with_weights=1 completeness=0.1 with_noise=0 kernel=gaussianquaternion



# # # with noise

# python train_sacred.py with device=1 loss=mmd log_name=new_exp lr=.01 freq_eval=10 num_particles=10 with_weights=1 completeness=0.1 with_noise=1 kernel=gaussian
# python train_sacred.py with device=0 loss=mmd log_name=new_exp lr=.01 freq_eval=10 num_particles=10 with_weights=1 completeness=0.1 with_noise=1 kernel=laplacequaternion
# python train_sacred.py with device=1 loss=mmd log_name=new_exp lr=.01 freq_eval=10 num_particles=10 with_weights=1 completeness=0.1 with_noise=1 kernel=gaussianquaternion


#python -m ipdb train.py --device=0 --loss=sinkhorn --log_name=new_exp --lr=.01 --freq_eval=10 --num_particles=10 --with_weights=0 --completeness=0.1 --with_noise=0 --kernel=laplacequaternion



#python train_sacred.py with device=0 loss=sinkhorn log_name=new_exp lr=.01 freq_eval=10 num_particles=10 with_weights=0 completeness=0.1 with_noise=0 kernel=gaussian total_iters=100000 &
python train_sacred.py with device=0 loss=sinkhorn log_name=new_exp lr=.01 freq_eval=10 num_particles=10 with_weights=0 completeness=0.1 with_noise=0 kernel=squared_euclidean &
python train_sacred.py with device=0 loss=sinkhorn log_name=new_exp lr=.01 freq_eval=10 num_particles=10 with_weights=0 completeness=0.1 kernel=power_quaternion &
#python train_sacred.py with device=0 loss=sinkhorn log_name=new_exp lr=.01 freq_eval=10 num_particles=10 with_weights=0 completeness=0.1 with_noise=0 kernel=sinkhorn_gaussian &
#python train_sacred.py with device=1 loss=mmd log_name=new_exp lr=.01 noise_level=0.05 freq_eval=10 num_particles=10 with_weights=0 completeness=0.1 with_noise=1 kernel=gaussian noise_decay_freq=1000  noise_decay=0.9 total_iters=100000 &


#python train.py  loss=sinkhorn --kernel=laplacequaternion


#python -m ipdb train.py --device=1 --loss=sinkhorn --log_name=new_exp --lr=.01 --freq_eval=10 --num_particles=10 --with_weights=0 --completeness=0.1 --with_noise=0 --kernel=laplacequaternion