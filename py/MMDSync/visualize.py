
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
#%matplotlib inline
from sklearn import metrics
import seaborn as sns
import itertools
import os
import pickle
sns.set(context='paper', style='whitegrid', font_scale=1.75)



def plot_from_dict(ax, xaxis, value, path_dict, color_dict,refresh=False):
	res_dicts = get_res(path_dict,refresh=refresh)
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
	values = ['eval_dist','loss','iteration','time']
	for value in values:
		tmp = [res_dict[value] for res_dict in res_dicts]
		out_dict[value] = np.stack(tmp,axis = 0)
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




