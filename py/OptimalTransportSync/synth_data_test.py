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
from synth_data_utils import *
from scipy.spatial.distance import cdist
from sklearn import mixture


if __name__ == "__main__":
    N=10
    completeness=1.0
    maxNumModes=4
    numParticles = 6
    
    print ('Generating synthetic data. N=' +str(N)+ ', #modes='+ str(maxNumModes) +', numParticles=' + str(numParticles))
    
    synthData = create_synth_graph(N,numParticles,completeness,maxNumModes)
    plot_synth_data(synthData)
        
    print ('This is one way to initialize the solutions by the random perturbations of the original')
    # initialize solution randomly
    N = synthData["N"]
    numParticles = synthData["numParticles"]
    sol = []
    solY = []
    for i in range(N):
        # sol{i} = np.random.randn(numParticles) # in order to generate truely random stuff
        sol.append(synthData["Xs"][i]+0.075*np.random.randn(numParticles))
        solY.append(np.ones((numParticles, 1))/numParticles) # initialize uniform weights