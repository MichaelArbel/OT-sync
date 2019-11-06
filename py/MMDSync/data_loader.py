import numpy as np
import torch
from torchvision import datasets
from torch.autograd import Variable
from torch.utils.data.sampler import SubsetRandomSampler
import torchvision
import torchvision.transforms as transforms
import os


def dataloader( data_path,name):

	edges_path = os.path.join(data_path,name+'_Edges.txt')
	edges = genfromtxt(edges_path, delimiter=',')

	Qabs_path = os.path.join(data_path,name+'_Qabs.txt')
	Qabs = genfromtxt(Qabs_path, delimiter=',')

	Qrel = os.path.join(data_path,name+'_Qrel.txt')
	Qrel = genfromtxt(Qrel, delimiter=',')

	vals = np.ones(edges.shape[1], dtype=np.double)
	rows = (np.asarray(edges[0,:])).astype(int)
	cols = (np.asarray(edges[1,:])).astype(int)
	set_edges = set(edges[0,:])
	set_edges.update(edges[1,:])
	N = len(set_edges)
	G = np.zeros((N,N))
	G[rows, cols] = vals

	C = nx.from_numpy_matrix(G)
	G = nx.from_numpy_matrix(G, create_using=nx.DiGraph())
	#I = np.array(list(G.edges()))

	if nx.is_connected(C):
		return edges, G, Qrel, Qabs 