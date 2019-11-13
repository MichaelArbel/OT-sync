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

def data_loader_notredame( data_path,name, dtype,device):

	edges_path = os.path.join(data_path,name+'_Edges.txt')
	edges = np.genfromtxt(edges_path, delimiter=',')
	edges = (np.array(edges)-1).astype('int64')

	Qabs_path = os.path.join(data_path,name+'_Qabs.txt')
	Qabs = np.genfromtxt(Qabs_path, delimiter=',')

	Qrel_path = os.path.join(data_path,name+'_Qrel.txt')
	Qrel = np.genfromtxt(Qrel_path, delimiter=',')

	Qrel = tr.tensor(Qrel, dtype=dtype,device=device).unsqueeze(1)    # do not assigne to gpu for now (this matrix can be huge)
	Qabs =  tr.tensor(Qabs, dtype=dtype, device=device).unsqueeze(1)
	Qabs[:,:,1:]*=-1.


	G,C = make_graphs(edges)



	#I = np.array(list(G.edges()))
	degree = np.array((G.degree()))
	max_index = np.argmax(degree[:,1])

	G, edges, Qabs = swap_nodes(G,edges,Qabs,0,max_index)


	# Additional formatting  : N x P x d 
	N,P,_ = Qabs.shape
	Nrel, Prel, _ = Qrel.shape
	# rotate all absolute poses so that the first camera becomes the reference
	Qabs = utils.quaternion_X_times_Y_inv(Qabs,Qabs[0,:,:].unsqueeze(0).repeat(N,1,1))
	wabs = (1./P) * tr.ones([N, P], dtype=dtype, device = device )
	wrel = (1./Prel) * tr.ones([Nrel, Prel], dtype=dtype, device = device )
	if nx.is_connected(C):
		return edges, G, Qrel, wrel, Qabs, wabs,None



def swap_nodes(G,edges,Qabs,node_1,node_2):
	successors_1 = [n for n in G.successors(node_1)]
	predecessors_1 = [n for n in G.predecessors(node_1)]
	successors_2 = [n for n in G.successors(node_2)]
	predecessors_2 = [n for n in G.predecessors(node_2)]	
	I = [tuple(l) for l in edges]

	K_s_1 = [I.index((node_1,s)) for s in successors_1]
	K_s_2 = [I.index((node_2,s)) for s in successors_2]
	K_p_1 = [I.index((p,node_1)) for p in predecessors_1]
	K_p_2 = [I.index((p,node_2)) for p in predecessors_2]
	edges[K_s_1,0] = node_2
	edges[K_s_2,0] = node_1
	edges[K_p_1,1] = node_2
	edges[K_p_2,1] = node_1
	tmp = Qabs[node_1,:,:]
	Qabs[node_1,:,:] = Qabs[node_2,:,:]
	Qabs[node_2,:,:] = tmp

	G,_=  make_graphs(edges)

	return G, edges, Qabs



def make_graphs(edges):	
	vals = np.ones(edges.shape[0], dtype=np.double)
	rows = (np.asarray(edges[:,0])).astype(int)
	cols = (np.asarray(edges[:,1])).astype(int)
	set_edges = set(edges[:,0])
	set_edges.update(edges[:,1])
	N = len(set_edges)
	GG = np.zeros((N,N))
	GG[rows, cols] = vals


	C = nx.from_numpy_matrix(GG)
	G = nx.from_numpy_matrix(GG, create_using=nx.DiGraph())
	return G,C


def data_loader_artsquad( data_path,name, dtype,device):

	edges_path = os.path.join(data_path,name+'_Edges.txt')
	edges = np.genfromtxt(edges_path, delimiter=',')
	edges = (np.array(edges)-2).astype('int64')

	Qabs_path = os.path.join(data_path,name+'_Qabs.txt')
	Qabs = np.genfromtxt(Qabs_path, delimiter=',')

	Qrel_path = os.path.join(data_path,name+'_Qrel.txt')
	Qrel = np.genfromtxt(Qrel_path, delimiter=',')

	Qrel = tr.tensor(Qrel, dtype=dtype,device=device).unsqueeze(1)    # do not assigne to gpu for now (this matrix can be huge)
	Qabs =  tr.tensor(Qabs, dtype=dtype, device=device).unsqueeze(1)
	Qabs = Qabs[1:,:,:]
	Qabs[:,:,1:]*=-1.
	
	vals = np.ones(edges.shape[0], dtype=np.double)
	rows = (np.asarray(edges[:,0])).astype(int)
	cols = (np.asarray(edges[:,1])).astype(int)
	N = Qabs.shape[0]
	G = np.zeros((N,N))
	G[rows, cols] = vals

	C = nx.from_numpy_matrix(G)
	G = nx.from_numpy_matrix(G, create_using=nx.DiGraph())
	
	connected_comp = list(nx.connected_components(C))
	#H = G.subgraph(connected_comp[0])
	#Qabs = Qabs[connected_comp[0],:,:]

	connected_comp =  list(connected_comp[0])
	indx_1 = tr.norm(Qabs,dim=-1)>0.5
	indx_1 = indx_1.int() 
	indx = tr.zeros([Qabs.shape[0],1,1]).int().to(device)
	indx[connected_comp,0]= 1
	indx = indx_1*indx
	indx= indx.to(device)
	indx = indx.bool()
	
	#I = np.array(list(G.edges()))



	# Additional formatting  : N x P x d 
	N,P,_ = Qabs.shape
	Nrel, Prel, _ = Qrel.shape
	# rotate all absolute poses so that the first camera becomes the reference
	Qabs = utils.quaternion_X_times_Y_inv(Qabs,Qabs[0,:,:].unsqueeze(0).repeat(N,1,1))
	wabs = (1./P) * tr.ones([N, P], dtype=dtype, device = device )
	wrel = (1./Prel) * tr.ones([Nrel, Prel], dtype=dtype, device = device )
	
	#if nx.is_connected(C):
	return edges, G, Qrel, wrel, Qabs, wabs,indx




