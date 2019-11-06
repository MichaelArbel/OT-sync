import numpy as np
import torch as tr
from torchvision import datasets
from torch.autograd import Variable
from torch.utils.data.sampler import SubsetRandomSampler
import torchvision
import torchvision.transforms as transforms
import os
import utils
import networkx as nx

def data_loader( data_path,name, dtype,device):

	edges_path = os.path.join(data_path,name+'_Edges.txt')
	edges = np.genfromtxt(edges_path, delimiter=',')
	edges = (np.array(edges)-1).astype(int)

	Qabs_path = os.path.join(data_path,name+'_Qabs.txt')
	Qabs = np.genfromtxt(Qabs_path, delimiter=',')

	Qrel_path = os.path.join(data_path,name+'_Qrel.txt')
	Qrel = np.genfromtxt(Qrel_path, delimiter=',')

	vals = np.ones(edges.shape[0], dtype=np.double)
	
	rows = (np.asarray(edges[:,0])).astype(int)
	cols = (np.asarray(edges[:,1])).astype(int)
	set_edges = set(edges[:,0])
	set_edges.update(edges[:,1])
	N = len(set_edges)
	G = np.zeros((N,N))
	G[rows, cols] = vals

	C = nx.from_numpy_matrix(G)
	G = nx.from_numpy_matrix(G, create_using=nx.DiGraph())
	#I = np.array(list(G.edges()))

	Qrel = tr.tensor(Qrel, dtype=dtype).unsqueeze(1)    # do not assigne to gpu for now (this matrix can be huge)
	Qabs =  tr.tensor(Qabs, dtype=dtype, device=device).unsqueeze(1)
	# Additional formatting  : N x P x d 
	N,P,_ = Qabs.shape
	Nrel, Prel, _ = Qrel.shape
	# rotate all absolute poses so that the first camera becomes the reference
	Qabs = utils.quaternion_X_times_Y_inv(Qabs,Qabs[0,:,:].unsqueeze(0).repeat(N,1,1))
	wabs = (1./P) * tr.ones([N, P], dtype=dtype, device = device )
	wrel = (1./Prel) * tr.ones([Nrel, Prel], dtype=dtype, device = device )
	if nx.is_connected(C):
		return edges, G, Qrel, wrel, Qabs, wabs




