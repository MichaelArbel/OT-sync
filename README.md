## Table of contents

* [Introduction](#introduction)
* [Requirements](#requirements)

* [Resources](#resources)
   * [Data](#data)
   * [Config files](#config-files)
   * [Hardware](#hardware)

* [How to use](#how-to-use)
* [Full documentation](#full-documentation)
* [Reference](#reference)
* [License](#license)

## Introduction

This repository contains an implementation of the algorithm for synchronizing probability measure on rotations using Optimal Transport and provides scripts to reproduce the results of the [paper](https://openaccess.thecvf.com/content_CVPR_2020/papers/Birdal_Synchronizing_Probability_Measures_on_Rotations_via_Optimal_Transport_CVPR_2020_paper.pdf) published at CVPR 2020.


## Requirements


This is a Pytorch implementation which requires the following packages:

```
python==3.6.2 or newer
torch==1.2.0 or newer
torchvision==0.4.0 or newer
numpy==1.17.2  or newer
geomloss==0.2.3 or newer
pandas==1.03 or newer
```

Main dependencies can be installed using:

```
pip install -r requirements.txt
```





## Resources

### Data

The data directory contains the following datasets : 'CastleP19', 'CastleP30', 'EntyP10', 'FountainP11', 'HerzJesuP25', 'HerzJesuP8'

### Config files
The config files to reproduce the main experiments in the paper are in ```core/configs/``` 


### Hardware

To use a particular GPU, set —device=#gpu_id
To use GPU without specifying a particular one, set —device=-1
To use CPU set —device=-2



## How to use


Go to ```core``` directory

```
cd core 
```
Set a log directory and specify a config file. For instance:
```
CONFIG_FILE='configs/CastleP19.yaml'
LOG_DIR='../logs'

```

Run the following command 

```
python train.py --config_data=$CONFIG_FILE --log_dir=$LOG_DIR
```
 


## Reference

If using this code for research purposes, please cite:

[1] T. Birdal,  M. Arbel 2 U. Simsekli, L. Guibas [*Synchronizing Probability Measures on Rotations via Optimal Transport*](https://openaccess.thecvf.com/content_CVPR_2020/papers/Birdal_Synchronizing_Probability_Measures_on_Rotations_via_Optimal_Transport_CVPR_2020_paper.pdf)

```
@inproceedings{birdal2020synchronizing,
  title={Synchronizing probability measures on rotations via optimal transport},
  author={Birdal, Tolga and Arbel, Michael and Simsekli, Umut and Guibas, Leonidas J},
  booktitle={Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition},
  pages={1569--1579},
  year={2020}
}                          }
```

## License 

This code is under a BSD license.