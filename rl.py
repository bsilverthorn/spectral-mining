from random import choice
import numpy as np
import scipy.sparse
import scipy.sparse.linalg
import spectral

def lstd(S, R, phi, lam=0.9):

    k = phi.shape[1] # number of features
    A = np.zeros((k,k))
    b = np.zeros((k,1))
    z = phi[S[0],:] # eligibility trace initialized to first state features
    for t in xrange(len(R)):
	    A = A + np.dot(z[:,None],(phi[S[t],:]-phi[S[t+1],:])[None,:])
	    b = b + R[t]*z[:,None]
	    z = lam*z+phi[S[t+1],:]

    beta = np.linalg.solve(A,b) # solve for feature parameters
    return beta

def room_walk(W,pos,v,g,eps=0.1,tmax=500):
    
    S = [pos]
    R = []
    t = 0
    while (pos != g) & (t < tmax):
	# choose next state
	if np.random.rand() < eps:
	    pos = choice(np.nonzero(W[:,pos])[0]) # random step 
        else:
	    pos = np.argmax(v[np.nonzero(W[:,pos])[0]]) # greedy step
	R.append(-1)
	S.append(pos)
	t += 1
    
    print 'number of steps to goal: ', t
    return S, R

def policy_iteration(num_iters = 10,num_eps= 100, wall_size=20):
   
    print "building adjacency matrix and laplacian"
    W = spectral.room_adjacency(wall_size) 
    L = spectral.laplacian_operator(W)
    #spL = scipy.sparse.csr_matrix(L)
    #(lam, v) = scipy.sparse.linalg.eigen_symmetric(splaplacian_NN, k = 8, which = "SM")
    print "solving for the eigenvectors"
    [eig_lam,eig_vec] = np.linalg.eigh(L)
    sort_inds = eig_lam.argsort()
    eig_lam = eig_lam[sort_inds]
    phi	 = eig_vec[:,sort_inds]

    print "performing policy iteration"

    k = phi.shape[1] # number of features
    phi = phi[:,:k]
    assert phi.shape[1] == k
    n = W.shape[0] # number of states

    beta = np.random.random((k,1)) # initialize feature weights
    v = np.dot(phi,beta) # set value according to current weights, features

    g = wall_size + 1 # goal is a position near the corner, not on a wall
    epsilons = [0.4]*num_eps
    #epsilons =0.5*np.exp(-np.array(range(num_eps))/(num_eps/2.)) 
    for i in xrange(num_iters):
	print "iteration: ", i
	# collect num_eps episodes of data with current policy
	for e in xrange(num_eps): 
            init_pos = np.round(np.random.random()*(n-1)) # start from random position
            if e == 0:
                S,R = room_walk(W,init_pos,v,g,epsilons[e]) # collect state transitions with eps-greedy walk
            else:
                _S,_R = room_walk(W,init_pos,v,g,epsilons[e])
                S.extend(_S); R.extend(_R)
	beta = lstd(S,R,phi) # get new feature weights
	print phi.shape
	print beta.shape
	v = np.dot(phi,beta)
    
    G = np.zeros((n,1)); G[g] = 1
    spectral.plot_functions_on_room(wall_size,np.hstack((G,v)))

def main():
    policy_iteration()


if __name__ == '__main__':
    main()
            
	   




