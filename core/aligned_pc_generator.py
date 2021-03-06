from open3d import *
import time
import numpy as np
from subprocess import STDOUT, check_output
from open3d import *
import copy
import uuid
import random
import shutil
import os
from os.path import isdir, join, exists
import pdb
import scipy
from scipy import io


if __name__ == "__main__":
	#scannet_dir = '/u/yifan/scannet'
	scannet_dir = 'D:/Data/meas_sync/kbest/ShapeNet2Samples/model1/8e15e57014a36199e52028751701a83'
	point_cloud_dir = join(scannet_dir, 'point_cloud')
	print(point_cloud_dir)
	dirs_by_object = [f for f in os.listdir(point_cloud_dir) if isdir(join(point_cloud_dir, f))]
	for object_dir in dirs_by_object:
		object_dir_path = join(point_cloud_dir, object_dir)
		cur_point_cloud_dir_path = object_dir_path
		cur_4pcs_output_dir_path = join(scannet_dir, 'globalreg_output', object_dir)
		obj_files = [f for f in os.listdir(cur_point_cloud_dir_path) if f.endswith('.mat')]
		obj_files.sort()
		for i in range(30):
			obj_i = join(cur_point_cloud_dir_path, obj_files[i])
			for j in range(i+1, 30):
				obj_j = join(cur_point_cloud_dir_path, obj_files[j])
				rot_mat_name = obj_files[i][:-4] + '_to_' + obj_files[j][:-4] + '.txt'
				rot_obj_name = obj_files[i][:-4] + '_to_' + obj_files[j][:-4] + '.obj'
				rot_mat_path = join(cur_4pcs_output_dir_path, rot_mat_name)
				rot_obj_path = join(cur_4pcs_output_dir_path, rot_obj_name)
				#if exists(rot_mat_path) or exists(rot_obj_path): continue
				pc_src = scipy.io.loadmat(obj_i)
				pc_src = pc_src['vertex']
				pc_src = pc_src.T
				#pc_src = np.loadtxt(obj_i, usecols=range(1,4))
				#pc_tgt = np.loadtxt(obj_j, usecols=range(1,4))
				ret = np.loadtxt(rot_mat_path)
				pc_src = np.concatenate((pc_src, np.ones((pc_src.shape[0], 1))), axis=1)
				print(ret)
				pc_src = np.dot(ret, pc_src.T)
				with open(rot_obj_path, 'w') as f:
					for k in range(pc_src.shape[1]):
						print('v {:.4f} {:.4f} {:.4f}'.format(pc_src[0,k], pc_src[1,k], pc_src[2,k]), file=f)