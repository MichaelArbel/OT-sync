import struct
import random
import math
import numpy as np
import scipy
import networkx as nx
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import matplotlib.pylab as pl
import matplotlib.animation as animation
from scipy.spatial.distance import cdist
from sklearn import mixture

def create_synth_graph(N,numParticles=16,completeness=1,maxNumModes=4):
    
    G = nx.Graph()
    Npair = int(N*(N-1)/2)
    I = np.zeros((2, Npair))

    # generate all pairwise edges
    k=0
    for i in range(N):
        for j in range(i,N):
            if(i!=j):
                I[:,k]=[i,j]
                k=k+1

    # now keep a portion of the edges
    e = math.ceil(completeness*Npair)
    ind = np.random.choice(Npair, e,  replace=False)
    I = I[:, ind]
    vals = np.ones(e, dtype=np.double)
    rows = (np.asarray(I[0,:])).astype(int)
    cols = (np.asarray(I[1,:])).astype(int)

    G = np.zeros((N,N))
    G[rows, cols] = vals
    
    e = I.shape[1]

    Xs = []
    Ys = []
    YsWeight = []
    DistrGndTruth = []
    Ratios = []
    RatiosY = []
    RatiosYWeight = []

    # generate emprical prior distributions for each node
    for i in range(0, N):
    
        numModes = np.random.randint(maxNumModes)+1
        mus = np.random.randn(numModes)
        vars = np.random.rand(numModes)*0.3
        covs = np.diag(vars)
        
        Y = np.ones((numParticles))/numParticles
        X = np.ones((numParticles))
        
        k = 0
        numPtsPerMode = np.int(numParticles/maxNumModes)
        for j in range(maxNumModes):
            xcur = np.random.normal(loc=0.0, scale=1.0,size=numPtsPerMode)
            #if j < numModes:

            #    X[k:k+numPtsPerMode] = xcur*vars[j] + mus[j]
            #else:
            X[k:k+numPtsPerMode] = xcur

            k=k+numPtsPerMode
        
        Xs.append(X)
        Ys.append(Y)
        YsWeight.append(Y)
        DistrGndTruth.append(X)

    
    for k in range(e):
        i = np.int(I[0,k])
        j = np.int(I[1,k])
        xi = Xs[i]
        xj = Xs[j]
        yi = np.asmatrix(Ys[i])
        yj = np.asmatrix(Ys[j])
        ywi = np.asmatrix(YsWeight[i])
        ywj = np.asmatrix(YsWeight[j])
        
        # compute the cost matrix
        
        r = cdist(xi.reshape(-1,1),xj.reshape(-1,1))
        r = np.reshape(r,r.shape[0]*r.shape[1],1)
        y = np.matmul(yi.T,yj)
        yw = np.matmul( ywi.T,ywj)
        y = np.reshape(y,y.shape[0]*y.shape[1],1)
        yw = np.reshape(yw,yw.shape[0]*yw.shape[1],1)
        y = y/np.sum(y)
        yw = yw/np.sum(yw);
        Ratios.append(r)
        RatiosY.append(y)
        yw = np.reshape(yw,[yw.shape[1],1])
        yw = np.squeeze(np.asarray(yw))
        RatiosYWeight.append(yw)
        
    
    synthData = {"type":"1d-gaussian", "N":N, "numParticles":numParticles, "numModes":numModes, "numEdges":e, "completeness":completeness,"G":G, "I":I,
                 "Xs":Xs,"Ys":Ys,"YsWeight":YsWeight, "DistrGndTruth":DistrGndTruth, "Ratios":Ratios, "RatiosY":RatiosY,"RatiosYWeight":RatiosYWeight}

    return synthData

def plot_synth_data(synthData):
    e = synthData["numEdges"]
    I = synthData["I"]
    Xs = synthData["Xs"]
    Ys = synthData["Ys"]
    Ratios = synthData["Ratios"]
    RatiosYWeight = synthData["RatiosYWeight"]
    
    numPlots = 4
    fig, axes = plt.subplots(nrows=numPlots, ncols=3, sharex=True, sharey=True)
        
    cnt = 0
    for k in range(0,e,int(e/(numPlots-1))):
        i = np.int(I[0,k])
        j = np.int(I[1,k])
        xi = Xs[i]
        xj = Xs[j]
        yi = (Ys[i])
        yj = (Ys[j])       
        r = Ratios[k]
        weightsR = RatiosYWeight[k]
        
        axes[cnt, 0].stem(xi,yi)
        axes[cnt, 1].stem(xj,yj)
        axes[cnt, 2].stem(r,weightsR)
        cnt= cnt+1
    
    fig.text(0.5, 0.02, 'Samples Xi            Samples Xj            Samples Xij', ha='center')
    fig.text(0.04, 0.5, 'Weights', va='center', rotation='vertical')
    fig.show()
    plt.show()





