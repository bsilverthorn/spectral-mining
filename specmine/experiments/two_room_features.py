import numpy
import math
import csv
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from matplotlib import cm
import specmine

def expand_columns(v,n):
    ''' adds zeros to the column vectors of matrix v at all walls for plotting '''
    n_cols = v.shape[1]
    V = numpy.vstack((v[0:n*(n/2),:],numpy.zeros([n/2,n_cols]),v[n*(n/2),:],numpy.zeros([(n-1)-n/2,n_cols]),v[n*(n/2)+1:,:]))
    return V

def two_room_adjacency(size):
    ''' create adjacency matrix '''

    adjacents = numpy.array([[-1,0],[1,0],[0,-1],[0,1]])
    adjacent_indxs = [-1,1,-size,size]

    
    W = numpy.zeros((size*size,size*size))
    for i in xrange(size*size):
        pos = numpy.array([i%size,math.floor(i/size)])
        
        # if the position being checked is not inside a wall
        if (pos[0] == size/2) | (pos[1] != size/2):
            for j in xrange(len(adjacents)):
                pos_adj = pos + adjacents[j]
                # if the adjacent position being checked is inside the grid
                if (pos_adj[0] > -1) & (pos_adj[0] < size) & (pos_adj[1] > -1) & (pos_adj[1] < size):
                    # if the adjacent position is not inside a wall
                    if (pos_adj[0] == size/2) | (pos_adj[1] != size/2):
                        W[i+adjacent_indxs[j],i] = 1

    # remove unvisitable states
    d = numpy.sum(W,1)
    keep = d.nonzero()[0]
    W = W[keep,:]
    W = W[:,keep]

    return W

def main(num_eigs = 10, size = 20):
    
    W = two_room_adjacency(size)
    basis = specmine.spectral.laplacian_basis(W,k = num_eigs)
    
    fig = plt.figure()
    #ax = fig.add_subplot(111, projection='3d')
    ax = Axes3D(fig)

    v = basis[:,:num_eigs+1]
    # add walls (with zero values) back to eigenvectors before plotting 
    V = expand_columns(v,size)

    w = csv.writer(file(specmine.util.static_path( \
        'two_room_features.csv'),'wb'))
    w.writerow(['eigen_num', 'x', 'y', 'value'])
    for e in xrange(num_eigs):
        Z = numpy.reshape(V[:,e],[size,size])
        for i in xrange(size):
            for j in xrange(size):
                w.writerow([e,i,j,Z[i,j]])
#    [X,Y] = numpy.meshgrid(range(size),range(size))
#    ax.plot_surface(X, Y, Z, rstride=1, cstride=1, cmap=cm.jet, linewidth=0, antialiased=False)
#    plt.show()

if __name__ == '__main__':
    main()
