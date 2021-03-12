
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
def set_axis_prop(ax, titles,sub_titles,scale_x='log',scale_y='log',xlabel='Iterations'):
    if len(ax.shape)==1:
        ax =np.array([ax])
    for i in range(len(ax)):
        for j in range(len(ax[0,:])):
            ax[i,j].set_title(sub_titles[j] +', ' + titles[i])
            ax[i,j].set_xscale(scale_x)
            ax[i,j].set_yscale(scale_y)
            ax[i,j].set_ylabel('$Error$')
            ax[i,j].set_xlabel(xlabel)

def make_color_dic(res_path):
	colors = sns.color_palette("Set1", n_colors=len(res_path), desat=.7)
	color_dict ={}
	for i, key in enumerate(res_path.keys()):
		color_dict[key] = colors[i]
	return color_dict

def unique(list1): 
      
    # insert the list to the set 
    list_set = set(list1) 
    # convert the set to the list 
    unique_list = (list(list_set)) 
    return unique_list
def get_selected_res_by(res_dicts,item_name):
    
	
	all_keys = list(res_dicts.keys())
	values_keys = list(res_dicts[all_keys[0]].keys())
	tmp_dict = {}
	for values_key in values_keys:
		tmp_dict[values_key] = []
	tmp_dict[item_name] = []
	key_list = []
	items_list = []

	for key in  all_keys:
		new_key, item_value = parse_key(key,item_name)
		key_list.append(new_key)
		items_list.append(item_value)
		for value_key in values_keys:
			val   = res_dicts[key][value_key][-1]
			tmp_dict[value_key].append(val)
		tmp_dict[item_name].append(item_value)

	idx_keys, ukeys =  get_unique(key_list)

	out_dict = {}

	for i ,ukey in enumerate(ukeys):
		out_dict[ukey] = {}
		for value_key in tmp_dict.keys():
			out_dict[ukey][value_key] = np.array(tmp_dict[value_key])[idx_keys[i]]

	return out_dict

def to_array(idices, items):
	return [np.array(items[idx]) for idx in indices]
	

def get_unique(keys):
	unique_keys = unique(keys)
	out =[]
	for ukey in unique_keys:
		indices = [i for i, x in enumerate(keys) if x == ukey]
		out.append(indices)

	return out, unique_keys


def parse_key(key_name,item_name):

	splits = key_name.split(' ')
	item_id = [i for i, x in enumerate(splits) if x.startswith(item_name)]

	item = splits.pop(item_id[0])
	
	item_value = float(item.split(':')[1])
	ss = ' '
	new_key = ss.join(splits)
	return new_key, item_value



def get_all_particles(path_dict,iteration):
	res_dict = {}
	for key, value in path_dict.items():
		pickle_in = open(os.path.join(value,"data/iter_"+str(iteration)+".pickle"),"rb")
		data = pickle.load(pickle_in)
		res_dict[key] = data
	return res_dict

def get_dicts(exp_path,subset_dictionary):
	dirs = os.listdir(exp_path) 
	directory = dirs[0]
	dicts = parse_dir(directory)
	dicts_from_dirs = [parse_dir(directory) for directory in dirs]
	dirs = [(dict_to_name(minus_dict(subset_dictionary,dictionary)),exp_path+directory) for directory, dictionary in zip(dirs,dicts_from_dirs) if  is_subset_dict(subset_dictionary,dictionary)]
	return dict(dirs)

def get_dicts_by(item,exp_path,subset_dictionary):
	dirs = os.listdir(exp_path) 
	directory = dirs[0]
	dicts = parse_dir(directory)
	dicts_from_dirs = [parse_dir(directory) for directory in dirs]
	dirs = [(dict_to_name(minus_dict(subset_dictionary,dictionary)),exp_path+directory) for directory, dictionary in zip(dirs,dicts_from_dirs) if  is_subset_dict(subset_dictionary,dictionary)]
	return dict(dirs)

def parse_dir(directory):
	splits = directory.split('_')
	dicts = {splits[2*i+1]:splits[2*(i+1)] for i in range(int(len(splits)/4)) }
	j = 10
	dicts_2 = {splits[2*i+j]:splits[2*i+1+j] for i in range(int(len(splits)/4)) }
	dicts.update(dicts_2)
	return dicts
def minus_dict(subset_dictionary,dictionary):
	all(map( dictionary.pop, subset_dictionary))
	return dictionary

def is_subset_dict(subset_dictionary,dictionary ):
	return set(subset_dictionary.items()).issubset( set(dictionary.items()) )

def dict_to_name(dictionary):
	s = ' '
	name = s.join(['%s:%s' % (key, value) for (key, value) in dictionary.items()])
	return name

def get_selected_res(exp_path,subset_dictionary):
	res_path =get_dicts(exp_path,subset_dictionary)
	colors = sns.color_palette("Set1", n_colors=len(res_path), desat=.7)
	color_dict ={}
	for i, key in enumerate(res_path.keys()):
		color_dict[key] = colors[i]
	refresh = True
	res_dicts = get_res(res_path,refresh=refresh)
	return res_dicts,color_dict


def make_color_dic(res_path):
	colors = sns.color_palette("Set1", n_colors=len(res_path), desat=.7)
	color_dict ={}
	for i, key in enumerate(res_path.keys()):
		color_dict[key] = colors[i]
	return color_dict





def plot_from_dict(ax,res_dicts, xaxis, value, color_dict,sort=False,lw=2.):
	#res_dicts = get_res(path_dict,refresh=refresh)

		for key in res_dicts.keys():
			if xaxis=='time':

				ax.plot(res_dicts[key][xaxis]-res_dicts[key][xaxis][0],res_dicts[key][value], lw=lw, label=key,color = color_dict[key])
			else:
				if 	sort:
					order = np.argsort(res_dicts[key][xaxis])
					xs = np.array(res_dicts[key][xaxis])[order]
					ys = np.array(res_dicts[key][value])[order]
					ax.plot(xs,ys, lw=lw, label=key,color = color_dict[key])
				else:
					ax.plot(res_dicts[key][xaxis],res_dicts[key][value], lw=lw, label=key,color = color_dict[key])
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
		if out_dict[key] is None:
			del out_dict[key]
	return out_dict

def stack_all_res(res_dicts):
	out_dict = {}
	values = ['eval_RM_dist','eval_dist','loss','iteration','time','avg_min_dist','median_min_dist','mode_weights']
	for value in values:
		tmp = [res_dict[value] for res_dict in res_dicts]
		out_dict[value] = np.stack(tmp,axis = 0)

	return out_dict

# def stack_all_res(res_dicts):

# 	out_dict = {}
# 	values = ['eval_RM_dist','eval_dist','loss','iteration','time']
# 	for value in values:
# 		tmp = [res_dict[value] for res_dict in res_dicts]
# 		out_dict[value] = np.stack(tmp,axis = 0)
# 	values = ['avg_min_dist','median_min_dist','mode_weights']
# 	try:
# 		for value in values:
# 			tmp = [res_dict[value] for res_dict in res_dicts]
# 			out_dict[value] = np.stack(tmp,axis = 0)
# 	except:
# 		#import pdb
# 		#pdb.set_trace()
# 		Num_iter = len(res_dicts)
# 		GT_quaternions = tr.from_numpy(res_dicts[0]['true_particles'][1:,:,:])
# 		avg_min_dist = []
# 		median_min_dist = []
# 		for res_dict in res_dicts:
# 			#out_dict = get_all_particles(res_path,iteration)
# 			quaternions = tr.from_numpy(res_dict['particles'][1:,:,:])
# 			dist = utils.quaternion_geodesic_distance(GT_quaternions,quaternions)
# 			min_dist,_ = tr.min(dist,dim=-1)
# 			avg_best_dist = tr.mean(min_dist,dim=-1)

# 			median_min_dist.append( np.median(avg_best_dist[1:].detach().cpu().numpy()))
# 			avg_min_dist.append(tr.mean(avg_best_dist[1:]).item())

# 		out_dict['median_min_dist'] = np.stack(median_min_dist,axis = 0)
# 		out_dict['avg_min_dist'] = np.stack(avg_min_dist,axis = 0)
# 		out_dict['mode_weights'] = np.stack(avg_min_dist,axis = 0)

# 	return out_dict


def make_res(value):
	res_dicts = get_all_iter(os.path.join(value,"data"))
	if res_dicts is not None:
		out_res = stack_all_res(res_dicts)
		with open(os.path.join(value, "data/stacked_res.pickle"),"wb") as pickle_out:
			pickle.dump(out_res, pickle_out)
		return out_res
	else:
		return None


def get_all_iter(main_dir):
	try:
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
	except:
		return None

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



