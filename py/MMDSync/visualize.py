
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
#%matplotlib inline
from sklearn import metrics
import seaborn as sns
import itertools
import os
import pickle
import torch as tr
#import matlab.engine
#import plot_help as db
import cv2
import utils
sns.set(context='paper', style='whitegrid', font_scale=1.75)



def plot_from_dict(ax,res_dicts, xaxis, value, color_dict):
	#res_dicts = get_res(path_dict,refresh=refresh)
	for key in res_dicts.keys():
		if xaxis=='time':

			ax.plot(res_dicts[key][xaxis]-res_dicts[key][xaxis][0],res_dicts[key][value], lw=2., label=key,color = color_dict[key])
		else:
			ax.plot(res_dicts[key][xaxis],res_dicts[key][value], lw=2., label=key,color = color_dict[key])
def get_res(path_dict,refresh=False):
	out_dict = {}
	for key, value in path_dict.items():
		if not refresh:
			try:
				pickle_in = open(os.path.join(value,"data/stacked_res.pickle"),"rb")
				out_dict[key] = pickle.load(pickle_in)
			except:
				out_dict[key] = make_res(value)
		else:
			out_dict[key] = make_res(value)
	return out_dict

def stack_all_res(res_dicts):
	out_dict = {}
	values = ['eval_RM_dist','eval_dist','loss','iteration','time','avg_min_dist','median_min_dist']
	for value in values:
		tmp = [res_dict[value] for res_dict in res_dicts]
		out_dict[value] = np.stack(tmp,axis = 0)
	return out_dict

def stack_all_res(res_dicts):

	out_dict = {}
	values = ['eval_RM_dist','eval_dist','loss','iteration','time']
	for value in values:
		tmp = [res_dict[value] for res_dict in res_dicts]
		out_dict[value] = np.stack(tmp,axis = 0)
	values = ['avg_min_dist','median_min_dist']
	try:
		for value in values:
			tmp = [res_dict[value] for res_dict in res_dicts]
			out_dict[value] = np.stack(tmp,axis = 0)
	except:
		#import pdb
		#pdb.set_trace()
		Num_iter = len(res_dicts)
		GT_quaternions = tr.from_numpy(res_dicts[0]['true_particles'][1:,:,:])
		avg_min_dist = []
		median_min_dist = []
		for res_dict in res_dicts:
			#out_dict = get_all_particles(res_path,iteration)
			quaternions = tr.from_numpy(res_dict['particles'][1:,:,:])
			dist = utils.quaternion_geodesic_distance(GT_quaternions,quaternions)
			min_dist,_ = tr.min(dist,dim=-1)
			avg_best_dist = tr.mean(min_dist,dim=-1)

			median_min_dist.append( np.median(avg_best_dist[1:].detach().cpu().numpy()))
			avg_min_dist.append(tr.mean(avg_best_dist[1:]).item())

		out_dict['median_min_dist'] = np.stack(median_min_dist,axis = 0)
		out_dict['avg_min_dist'] = np.stack(avg_min_dist,axis = 0)

	return out_dict


def make_res(value):
	res_dicts = get_all_iter(os.path.join(value,"data"))
	out_res = stack_all_res(res_dicts)
	with open(os.path.join(value, "data/stacked_res.pickle"),"wb") as pickle_out:
		pickle.dump(out_res, pickle_out)
	return out_res


def get_all_iter(main_dir):
	files = [name for name in os.listdir(main_dir) if name.startswith('iter')]
	res_dicts = []
	for file in files:
		pickle_in = open(os.path.join(main_dir,file),"rb")
		res_dicts.append( pickle.load(pickle_in))
		pickle_in.close()
	pickle_out = open(os.path.join(main_dir, "all_res.pickle"),"wb")
	pickle.dump(res_dicts, pickle_out)
	pickle_out.close()
	return res_dicts

def get_all_particles(path_dict,iteration):
	res_dict = {}
	for key, value in path_dict.items():
		pickle_in = open(os.path.join(value,"data/iter_"+str(iteration)+".pickle"),"rb")
		data = pickle.load(pickle_in)
		res_dict[key] = data
	return res_dict


def make_bg_sphere(eng,particles,ground_truth=None,quality=50):
	particles = np.split(particles,particles.shape[0])
	particles = [matlab.double(p.tolist()) for p in particles]
	bingham_fits = []
	if not ground_truth is None:
		ground_truth = np.split(ground_truth,ground_truth.shape[0])
		#ground_truth = [matlab.double(p.tolist()) for p in ground_truth]
		for p, gp in zip(particles,ground_truth):
			bingham_fits.append(db.get_bingham(eng, p, GT=gp, precision=quality))
	else:
		for p in particles:
			bingham_fits.append(db.get_bingham(eng, p, GT=None, precision=quality))
	return bingham_fits
def make_bg_spheres_from_dict(eng,res_path,iterations,method,num_cameras,GT=False,quality=50):
	bg_spheres= []
	num_cameras +=1 
	if GT:
		out_dict = get_all_particles(res_path,0)
		GT_quaternions =  out_dict[method]['true_particles'][1:num_cameras,:,:]
	else:
		GT_quaternions =None
	for iteration in iterations:
		out_dict = get_all_particles(res_path,iteration)
		quaternions = out_dict[method]['particles'][1:num_cameras,:,:]
		bg_spheres.append(make_bg_sphere(eng,quaternions,ground_truth= GT_quaternions,quality=quality))
	bg_spheres = list(map(list, zip(*bg_spheres)))
	return bg_spheres
def concat_tile(im_list_2d):
	return cv2.vconcat([cv2.hconcat(im_list_h) for im_list_h in im_list_2d])



