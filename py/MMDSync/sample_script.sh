# no noise

python train.py --device=0 --loss=mmd --log_name=exp_gaussian --lr=.01 --freq_eval=10 --num_particles=10 --with_weights=0 --completeness=0.1 --with_noise=0 --kernel=gaussian &
python train.py --device=1 --loss=mmd --log_name=exp_laplace_quaternion --lr=.01 --freq_eval=10 --num_particles=10 --with_weights=0 --completeness=0.1 --with_noise=0 --kernel=laplacequaternion &
python train.py --device=0 --loss=mmd --log_name=exp_gaussian_quaternion --lr=.01 --freq_eval=10 --num_particles=10 --with_weights=0 --completeness=0.1 --with_noise=0 --kernel=gaussianquaternion &

