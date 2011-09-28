import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from matplotlib import cm
from sparseQR import sparseQR

def laplacian_operator(W):
    D = np.diag(np.sum(W,1))
    D_invsqrt = np.sqrt(np.linalg.inv(D))

    return np.dot(np.dot(D_invsqrt,(D-W)),D_invsqrt)

def diffusion_operator(W):
    D = np.diag(np.sum(W,1))
    D_invsqrt = np.sqrt(np.linalg.inv(D))

    return np.dot(np.dot(D_invsqrt,W),D_invsqrt)

def dw_tree(T,J,lam,p,eps_scal):
    Q_dict = {}
    print 'building diffusion wavelet tree'
    for j in range(J):
        print 'level j = ', j
        eps = np.sum(np.sum(T))/len(T[T>0])*eps_scal # heuristic for setting epsilon, from Bo Liu paper
        print 'epsilon: ', eps
        Q = sparseQR(T,eps,lam,p)
        print 'size of new basis: ',Q.shape
        Q_dict[j] = Q
        T = np.dot(np.dot(np.dot(Q.T,T),T),Q) # T^2 in k-space (kxk)

    return Q_dict

def expand_wavelets(Q_dict, k, n):
    J = len(Q_dict)
    Q_coarse = np.eye(n)
    for j in xrange(J):
        Q_coarse = np.dot(Q_coarse,Q_dict[j])
        if j == 0: 
            phi = Q_coarse
        # put the new (more coarse) column vectors on the left
        else:
            phi = np.hstack((Q_coarse,phi))

    # add the constant vector and return 
    return np.hstack((np.ones((n,1))/float(n),phi[:,:k-1]))

def build_diffusion_basis(T,k,J=8,lam=2.5,p=1,eps_scal=10**-3):
    n = T.shape[0] # number of states in complete space
    Q_dict = dw_tree(T,J,lam,p,eps_scal)

    return expand_wavelets(Q_dict,k,n)
    
    
def room_adjacency(n = 20,self_loops = True):
    adjacents = np.array([[-1,0],[1,0],[0,-1],[0,1]])
    adjacent_indxs = [-1,1,-n,n]

    # create adjacency matrix, W
    W = np.zeros([n**2,n**2])
    for i in range(n**2):
        pos = np.array([i%n,np.floor(i/n)])
        
        # if the position being checked is not inside a wall
        if (pos[0] == n/2) | (pos[1] != n/2):
            for j in range(len(adjacents)):
                pos_adj = pos + adjacents[j]
                # if the adjacent position being checked is inside the grid
                if (pos_adj[0] > -1) & (pos_adj[0] < n) & (pos_adj[1] > -1) & (pos_adj[1] < n):
                    # if the adjacent position is not inside a wall
                    if (pos_adj[0] == n/2) | (pos_adj[1] != n/2):
                        W[i+adjacent_indxs[j],i] = 1

    d = np.sum(W,1)
    # remove unvisitable states
    W = W[d>0,:]
    W = W[:,d>0]
    if self_loops: W = W+np.eye(W.shape[0]) # add self-transitions - makes not periodic?

    return W

def expand_columns(v,n):
    ''' adds zeros to the column vectors of matrix v at all walls for plotting '''
    n_cols = v.shape[1]
    V = np.vstack((v[0:n*(n/2),:],np.zeros([n/2,n_cols]),v[n*(n/2),:],np.zeros([(n-1)-n/2,n_cols]),v[n*(n/2)+1:,:]))
    return V

def plot_functions_on_room(n,F):
    
    V = expand_columns(F,n)
    [X,Y] = np.meshgrid(range(n),range(n))
    
    fig = plt.figure()
    if F.shape[1] == 1: rows = 1
    else: rows = 2
    cols = int(np.ceil(F.shape[1]/float(rows)))
    for i in range(F.shape[1]):
        ax = fig.add_subplot(rows,cols, i+1, projection='3d')
        Z = np.reshape(V[:,i],[n,n]) # uniformly sample which plots to show
        ax.plot_surface(X, Y, Z, rstride=1, cstride=1, cmap=cm.jet,linewidth=0, antialiased=True)
    plt.show()

