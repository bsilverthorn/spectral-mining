import plac
import numpy as np
import math
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from matplotlib import cm


def expand_columns(v):
    ''' adds zeros to the column vectors of matrix v at all walls for plotting '''
    V = np.vstack((v[0:50,:],np.zeros([5,91]),v[50,:],np.zeros([4,91]),v[51:,:]))
    return V


def laplacian_eigenvectors(eign_num):
    adjacents = np.array([[-1,0],[1,0],[0,-1],[0,1]])
    adjacent_indxs = [-1,1,-10,10]
    #walls = np.array([[5,0],[5,1],[5,2],[5,3],[5,4],[5,6],[5,7],[5,8],[5,9]])

    # create adjacency matrix
    W = np.zeros([100,100])
    for i in range(100):
        pos = np.array([i%10,math.floor(i/10)])
        
        # if the position being checked is not inside a wall
        if (pos[0] == 5) | (pos[1] != 5):
            for j in range(len(adjacents)):
                pos_adj = pos + adjacents[j]
                # if the adjacent position being checked is inside the grid
                if (pos_adj[0] > -1) & (pos_adj[0] < 10) & (pos_adj[1] > -1) & (pos_adj[1] < 10):
                    # if the adjacent position is not inside a wall
                    if (pos_adj[0] == 5) | (pos_adj[1] != 5):
                        W[i+adjacent_indxs[j],i] = 1

    d = np.zeros(W.shape[0])
    keep_inds = []
    for i in range(W.shape[0]):
        d[i] = np.sum(W[i,:])
        if d[i] != 0:
            keep_inds.append(i)
    D = np.diag(d)
    D = D[keep_inds,:] # remove unvisitable states 
    D = D[:,keep_inds]
    W = W[keep_inds,:]
    W = W[:,keep_inds]

    # normalized graph laplacian of adjacency graph
    D_invsqrt = np.sqrt(np.linalg.inv(D))
    L = np.dot(np.dot(D_invsqrt,(D-W)),D_invsqrt)
    
    
    ## combinatorial graph laplacian
    #L = D-W

    # find the eigenvalues/eigenvectors of the laplacian and sort by increasing eigenvalue
    [lam,v] = np.linalg.eigh(L)
    sort_inds = lam.argsort()
    lam = lam[sort_inds]
    v = v[:,sort_inds]
    
    fig = plt.figure()
    #ax = fig.add_subplot(111, projection='3d')
    ax = Axes3D(fig)


    # add walls (with zero values) back to eigenvectors before plotting 
    V = expand_columns(v)
    [X,Y] = np.meshgrid(range(10),range(10))
    Z = np.reshape(V[:,eign_num],[10,10])

    ax.plot_surface(X, Y, Z, rstride=1, cstride=1, cmap=cm.jet, linewidth=0, antialiased=False)

    plt.show()

@plac.annotations(
    eigen_index = ("eigenvalue index", "positional", None, int),
    )
def main(eigen_index):
    laplacian_eigenvectors(eigen_index)

if __name__ == "__main__":
    plac.call(main)

    
