from random import choice
import numpy as np
import scipy.sparse
import scipy.sparse.linalg
import spectral

def lstd_build(S, R, phi, lam=0.9):

    k = phi.shape[1] # number of features
    A = np.zeros((k,k))
    b = np.zeros((k,1))
    z = phi[S[0],:] # eligibility trace initialized to first state features
    for t in xrange(len(R)):
	    A += np.dot(z[:,None],(phi[S[t],:]-phi[S[t+1],:])[None,:])
	    b += R[t]*z[:,None]
	    z = lam*z+phi[S[t+1],:]

    return A, b

def lstd_solve(A,b):

    beta = np.linalg.solve(A,b) # solve for feature parameters
    return beta

def room_walk(W,pos,v,g,eps=0.1,tmax=300):
    
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
    return S, R, [t]

def policy_iteration(k = 20, num_iters = 3 ,num_eps = 300, wall_size=20):
   
    print "building adjacency matrix"
    W = spectral.room_adjacency(wall_size) 
    n = W.shape[0] # number of states
    
    #L = spectral.laplacian_operator(W)
    #spL = scipy.sparse.csr_matrix(L)
    #(lam, v) = scipy.sparse.linalg.eigen_symmetric(splaplacian_NN, k = 8, which = "SM")
    #print "solving for the eigenvectors"
    #[eig_lam,eig_vec] = np.linalg.eigh(L)
    #sort_inds = eig_lam.argsort()
    #eig_lam = eig_lam[sort_inds]
    #phi = eig_vec[:,sort_inds]
    #phi = phi[:,:k]

    print "building diffusion operator and basis"
    T = spectral.diffusion_operator(W)
    phi = spectral.build_diffusion_basis(T,k)
    print phi.shape    

    print "performing policy iteration"
    beta = np.random.random((k,1)) # initialize feature weights
    v = np.dot(phi,beta) # set value according to current weights, features

    g = 2*wall_size + 1 # goal is a position near the corner, not on a wall
    #init_pos = wall_size*(wall_size-1)
    epsilons = [0.2]*num_eps
    #epsilons =0.5*np.exp(-np.array(range(num_iters))/(num_iters/2.)) 
    for i in xrange(num_iters):
	print "iteration: ", i
	# collect num_eps episodes of data with current policy
        A = np.zeros((k,k))
        b = np.zeros((k,1)) 
        t = []
	for e in xrange(num_eps): 
            init_pos = np.round(np.random.random()*(n-1)) # start from random position
            S,R,_t = room_walk(W,init_pos,v,g,epsilons[i]) # collect state transitions with eps-greedy walk
            _A,_b = lstd_build(S,R,phi)
            A += _A; b += _b; t.extend(_t)
        print 'average time steps: ', np.mean(t) 
        beta = lstd_solve(A,b)
	v = np.dot(phi,beta)
    
    G = np.zeros((n,1)); G[g] = 1
    I = np.zeros((n,1)); I[init_pos] = 1
    #spectral.plot_functions_on_room(wall_size,np.hstack((G,I,v)))
    spectral.plot_functions_on_room(wall_size,v)

def main():
    policy_iteration()


if __name__ == '__main__':
    main()
            
	   




