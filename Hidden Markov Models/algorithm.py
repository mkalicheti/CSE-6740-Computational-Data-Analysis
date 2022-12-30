import matplotlib.pyplot as plt
import numpy as np

def algo(q, Y):
    # init
    p = 0.0
    n_w = len(Y) #number of weeks

    a = np.array([[0.8, 0.2], [0.2, 0.8]]) #transition matrix
    b = np.array([[q, 1-q], [1-q, q]]) #emission probabilities
    pi = [0.2, 0.8] #initial distribution probabilities

    offset  =  {-1:1, 1:0}

    #Forward Algo
    alpha = np.zeros((n_w, 2))
    #Initialisation
    alpha[0][0] = pi[0]*b[1][0] 
    alpha[0][1] = pi[1]*b[1][1]

    for w in range(2, n_w+1): #iteration
        alpha[w-1,0] = (alpha[w-2,0]*a[0,0]+alpha[w-2,1]*a[1,0])*b[offset[int(Y[w-1])],0]
        alpha[w-1,1] = (alpha[w-2,0]*a[0,1]+alpha[w-2,1]*a[1,1])*b[offset[int(Y[w-1])],1]    

    px = alpha[n_w-1,0]+alpha[n_w-1,1] #termination
    
    #Backward Algo
    beta = np.zeros((n_w, 2))
    beta[n_w-1, 0] = 1.0 #initialisation
    beta[n_w-1, 1] = 1.0

    for w in range(n_w-1, 0, -1): #iteration
        beta[w-1,0] = a[0, 0]*b[offset[int(Y[w])], 0]*beta[w, 0] + a[0, 1]*b[offset[int(Y[w])], 1]*beta[w, 1] 
        beta[w-1,1] = a[1, 0]*b[offset[int(Y[w])], 0]*beta[w, 0] + a[1, 1]*b[offset[int(Y[w])], 1]*beta[w, 1]

    p = alpha*beta/px #termination

    fig, ax = plt.subplots()
    ax.plot(p[:, 0])
    return p[n_w-1, 0], fig