import sys
import getopt
import numpy as np
import math
import matplotlib.pyplot as plt
import matplotlib as mpl
import cProfile
#import pstats
from sparseQR import sparseQR
#from mpl_toolkits.mplot3d.axes3d import Axes3D
from mpl_toolkits.mplot3d import axes3d, Axes3D
from matplotlib import cm

def expand_columns(v,n):
	''' adds zeros to the column vectors of matrix v at all walls for plotting '''
	n_cols = v.shape[1]
	# V = np.vstack((v[0:50,:],np.zeros([5,n_cols]),v[50,:],np.zeros([4,n_cols]),v[51:,:])) # n=10 case
	V = np.vstack((v[0:n*(n/2),:],np.zeros([n/2,n_cols]),v[n*(n/2),:],np.zeros([(n-1)-n/2,n_cols]),v[n*(n/2)+1:,:]))
	return V

def dw_tree(T,J,lam,p,eps_scal):
	Q_dict = {}
	for j in range(J):
		#Q,R = np.linalg.qr(T) # non-sparse QR (for comparison)
		eps = np.sum(np.sum(T))/len(T[T>0])*eps_scal # heuristic for setting epsilon, from Bo Liu paper
		print 'epsilon: ', eps
		Q = sparseQR(T,eps,lam,p) 
		print 'size of new basis: ',Q.shape
		Q_dict[j] = Q
		# representing T^2 in the original nxn space	
		#_T = np.dot(np.dot(np.dot(Q,np.dot(Q.T,T)),np.dot(T.T,Q)),Q.T) # if T.T = T
		#_T = np.dot(Q,np.dot(np.dot(np.dot(np.dot(Q.T,T),T),Q),Q.T)) # what is the most efficient order? (if Q = nxk and k < n)
		T = np.dot(np.dot(np.dot(Q.T,T),T),Q) # T^2 in k-space (kxk)
		
	return Q_dict

def plot_basis(Q,num_plots,n,j):
	
	V = expand_columns(Q,n)
	[X,Y] = np.meshgrid(range(n),range(n))
	
	fig = plt.figure()
	rows = max(min(3,Q.shape[1]/4),1)
	for i in range(num_plots):
		ax = fig.add_subplot(rows, round(num_plots/rows), i+1, projection='3d')
		Z = np.reshape(V[:,round(Q.shape[1]*i/num_plots)],[n,n]) # uniformly sample which plots to show
		#Z = np.reshape(V[:,i],[n,n]) # show only the "largest"
		ax.plot_surface(X, Y, Z, rstride=1, cstride=1, cmap=cm.jet,linewidth=0, antialiased=True)
	#save_str = 'Scaling_fncs_uniform_j='+str(j)+'.png'
	#plt.savefig(save_str)
	plt.show()
	

def diffusion_wavelets():
	
	num_plots = [12,12,12,12,9,9,9,9,6,6,4,2]
	J = 8 # number of levels of diffusion to compute (T^2^J)
	lam = 3 # slack between choosing the largest (L2) vector and min of L1
	p = 1
	eps_scal = 10**-3
	n = 20 # size of the room

	adjacents = np.array([[-1,0],[1,0],[0,-1],[0,1]])
	adjacent_indxs = [-1,1,-n,n]

	# create adjacency matrix, W
	W = np.zeros([n**2,n**2])
	for i in range(n**2):
		pos = np.array([i%n,math.floor(i/n)])
		
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
	W = W+np.eye(W.shape[0]) # add self-transitions - makes not periodic?

	# Set diffusion operator
	D = np.diag(np.sum(W,1))
	D_invsqrt = np.sqrt(np.linalg.inv(D))
	T = np.dot(np.dot(D_invsqrt,W),D_invsqrt) # normalized operator, or...
	#T = W # un-normalized
	
	# dyadic powers of T (for comparison)
#	T_2 = np.dot(T,T)
#	T_4 = np.dot(T_2,T_2)
#	T_8 = np.dot(T_4,T_4)
#	T_16 = np.dot(T_8,T_8)
#	T_32 = np.dot(T_16,T_16)
#	T_64 = np.dot(T_32,T_32)
#	T_128 = np.dot(T_64,T_64)
#	T_256 = np.dot(T_128,T_128)
	
	# create the diffusion wavelet tree
	Q_dict = dw_tree(T,J,lam,p,eps_scal)

	# expand most coarse bases back into full state space
	Q_coarse = np.eye(W.shape[0])
	for j in range(J):
		Q_coarse = np.dot(Q_coarse,Q_dict[j])
		
	plot_basis(Q_coarse,6,n,j)



def main():
	#cProfile.run('diffusion_wavelets()','dw_prof')
	#p = pstats.Stats('dw_prof')
	#p.strip_dirs().sort_stats('time').print_stats(20)
	diffusion_wavelets()

if __name__ == "__main__":
	main()
