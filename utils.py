import matplotlib.pyplot as plt 
import numpy as np

def plotLearning(scores, filename, x=None,xlabel = None, ylabel=None, window=5):   
    N = len(scores)
    running_avg = np.empty(N)
    for t in range(N):
	    running_avg[t] = np.mean(scores[max(0, t-window):(t+1)])
    
    if x is None:
        x = [i for i in range(N)]
    plt.ylabel(ylabel)       
    plt.xlabel(xlabel)                     
    plt.plot(x, running_avg)
    plt.plot(x, scores)
    plt.savefig(filename)
