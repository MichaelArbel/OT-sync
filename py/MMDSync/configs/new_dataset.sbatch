#!/bin/bash  
#SBATCH -p gpu  # partition (queue) 
#SBATCH -N 1 #number of nodes  
#SBATCH -n 1 # number of cores  
#SBATCH --mem=26G

#echo 'Hello world'

#sbatch -n 1 -t 1-14:00 -p gpu --mem=9G --gres=gpu:p5000:1 --wrap="python train.py --device=1 --log_in_file --config_method=configs/new_datasets --data_name='Madrid_Metropolis' "

python train.py --device=1 --log_in_file --config_data='configs/new_datasets.yaml' --data_name='Montreal_Notre_Dame' &
python train.py --device=0 --log_in_file --config_data='configs/new_datasets.yaml' --data_name='Madrid_Metropolis' &
python train.py --device=1 --log_in_file --config_data='configs/new_datasets.yaml' --data_name='Ellis_Island' &
python train.py --device=0 --log_in_file --config_data='configs/new_datasets.yaml' --data_name='Alamo' &
python train.py --device=0 --log_in_file --config_data='configs/new_datasets.yaml' --data_name='NYC_Library' &
python train.py --device=0 --log_in_file --config_data='configs/new_datasets.yaml' --data_name='Piazza_del_Popolo' &
python train.py --device=1 --log_in_file --config_data='configs/new_datasets.yaml' --data_name='Roman_Forum' &
python train.py --device=1 --log_in_file --config_data='configs/new_datasets.yaml' --data_name='Tower_of_London' &
python train.py --device=0 --log_in_file --config_data='configs/new_datasets.yaml' --data_name='Vienna_Cathedral' &
python train.py --device=0 --log_in_file --config_data='configs/new_datasets.yaml' --data_name='Yorkminster' &

python train.py --device=0 --log_in_file --config_data='configs/new_datasets.yaml' --data_name='Notre_Dame' &
