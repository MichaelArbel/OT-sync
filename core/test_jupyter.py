
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
#%matplotlib inline
import seaborn as sns
sns.set(context='paper', style='whitegrid', font_scale=1.75)
from visualize import *
#import plot_help as db
import cv2




# def get_dicts_by(item,exp_path,subset_dictionary):
#     dirs = os.listdir(exp_path) 
#     directory = dirs[0]
#     dicts = parse_dir(directory)
#     dicts_from_dirs = [parse_dir(directory) for directory in dirs]
#     dirs = [(dict_to_name(minus_dict(subset_dictionary,dictionary)),exp_path+directory) for directory, dictionary in zip(dirs,dicts_from_dirs) if  is_subset_dict(subset_dictionary,dictionary)]
#     return dict(dirs)

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








exp_path = '/nfs/data/michaela/projects/OptSync/synth_increase_noise/synth_increase_noise/'
subset_dictionary = {'N':'10','pow':'1.2','comp':'0.2','lr':'0.01'}
item_id = 3
res_dicts_1, color_dict_1 = get_selected_res(exp_path,subset_dictionary)
res_dicts_1 = get_selected_res_by(res_dicts_1,'sigma')

subset_dictionary = {'N':'10','pow':'1.2','comp':'0.5','lr':'0.01'}
#import pdb
#pdb.set_trace()
res_dicts_2, color_dict_2 = get_selected_res(exp_path,subset_dictionary)

#res_dicts_2 = get_selected_res_by(res_dicts_2,'sigma')

subset_dictionary = {'N':'10','pow':'1.2','comp':'0.7','lr':'0.01'}
res_dicts_3, color_dict_3 = get_selected_res(exp_path,subset_dictionary)
#res_dicts_3 = get_selected_res_by(res_dicts_3,'sigma')
all_res_dict = [res_dicts_1,res_dicts_2,res_dicts_3]
all_colors_dict = [color_dict_1,color_dict_2,color_dict_3]








