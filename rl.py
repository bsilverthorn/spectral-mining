import copy
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

def td_lambda(S,R,phi,lam=0.9, gamma = 1, beta = None, alpha = 0.01):
    k = phi.shape[1]
    if beta == None:
        beta = np.zeros((k,1))
    z = phi[S[0],:]
    for t in xrange(len(R)):
        curr_phi = phi[S[t],:] 
        delta = z[:,None]*(R[t]+np.dot((gamma*phi[S[t+1],:]-curr_phi)[None,:],beta))
        z = lam*z+phi[S[t+1],:]
        beta += delta*alpha/np.linalg.norm(curr_phi,1)

    return beta

def q_lambda(W,S,R,phi,lam=0.9, gamma = 1, beta = None, alpha = 0.01):
    k = phi.shape[1]
    if beta == None:
        beta = np.zeros((k,1))
    z = phi[S[0],:]
    v = np.dot(phi,beta)

    for t in xrange(len(R)):
        neighbors = np.nonzero(W[:,S[t]])[0]
        best_state = neighbors[np.argmax(v[neighbors])]
        curr_phi = phi[S[t],:] 
        delta = z[:,None]*(R[t]+np.dot((gamma*phi[best_state,:]-curr_phi)[None,:],beta))
        z = gamma*lam*z+phi[S[t+1],:]
        beta += delta*alpha/np.linalg.norm(curr_phi,1)
        v = np.dot(phi,beta)

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
            neighbors = np.nonzero(W[:,pos])[0]
            pos = neighbors[np.argmax(v[neighbors])] # greedy step

        if pos == g:
            R.append(100.)
        else:
            R.append(-0.1)
        S.append(pos)
        t += 1

    return S, R, t

def heat_map(S,n):
    
    h = np.zeros((n,1))
    for s in S:
        h[s] += 1

    return h # /float(len(S))

def lstd_room_policy_iter(W,phi,epsilons,num_iters,num_eps,n,g):
    
    print "performing policy iteration"
    k = phi.shape[1]
    beta = np.zeros((k,1))
    v = np.dot(phi,beta) # set value according to current weights, features
    A = np.zeros((k,k))
    b = np.zeros((k,1))
    h = np.zeros((n,1)) 
    init_pos = n-1
    for i in xrange(num_iters):
        print "iteration: ", i
        print 'epsilon: ' , epsilons[i] 
        # collect num_eps episodes of data with current policy
        #A = np.zeros((k,k))
        #b = np.zeros((k,1))
        #h = np.zeros((n,1)) 
        t = 0
        ep = 0
        while ep < num_eps: #for e in xrange(num_eps):  
            #init_pos = np.round(np.random.random()*(n-1)) # start from random position
            S,R,_t = room_walk(W,init_pos,v,g,epsilons[i]) # collect state transitions with eps-greedy walk
            if S[-1] == g:
                print ep
                ep += 1
                h += heat_map(S,n)
                _A,_b = lstd_build(S,R,phi)
                A += _A; b += _b; t += _t

        print 'average time steps: ', t/float(ep)
        beta = lstd_solve(A,b)
        v = np.dot(phi,beta)
    
    return beta,h, init_pos 

def td_room_policy_iter(W,phi,epsilons,num_iters,num_eps,n,g):
    print "performing policy iteration"
    k = phi.shape[1]
    weights = np.zeros((k,1))
    v = np.dot(phi,weights) # set value according to current weights, features

    init_pos = n-1
    for i in xrange(num_iters):
        print "iteration: ", i
        print 'epsilon: ' , epsilons[i] 
        # collect num_eps episodes of data with current policy
        h = np.zeros((n,1)) 
        t = 0
        ep = 0
        while ep < num_eps:
        #for e in xrange(num_eps):
            #init_pos = np.round(np.random.random()*(n-1)) # start from random position
            S,R,_t = room_walk(W,init_pos,v,g,epsilons[i]) # collect state transitions with eps-greedy walk
            if S[-1] == g:
                print ep
                ep += 1
                h += heat_map(S,n)  
                t += _t              
                #weights = td_lambda(S,R,phi,beta = weights)
                weights = q_lambda(W,S,R,phi,beta = weights)

        print 'average time steps: ', t/float(num_eps)
        v = np.dot(phi,weights)  

    return weights, h, init_pos

def policy_iteration(k = 100, num_iters = 5 ,num_eps = 20, wall_size=20):
   
    print "building adjacency matrix"
    W = spectral.room_adjacency(wall_size, False) 
    n = W.shape[0] # number of states
    
    L = spectral.laplacian_operator(spectral.room_adjacency(wall_size))
    #spL = scipy.sparse.csr_matrix(L)
    
    print "solving for the eigenvectors"
    #(eig_lam, eig_vec) = scipy.sparse.linalg.eigen_symmetric(spL, k, which = "SM")
    [eig_lam,eig_vec] = np.linalg.eigh(L)
    sort_inds = eig_lam.argsort()
    eig_lam = eig_lam[sort_inds]
    phi = eig_vec[:,sort_inds]
    phi = phi[:,:k]

    #print "building diffusion operator and basis"
    #T = spectral.diffusion_operator(W)
    #phi = spectral.build_diffusion_basis(T,k)

    
    g = 2*wall_size + 1 # goal is a position near the corner, not on a wall
    epsilons = [1]*num_eps
    #epsilons = 0.5*np.exp(-np.array(range(num_iters))/(num_iters/2.)) 
    #beta,h,init_pos = lstd_room_policy_iter(W,phi,epsilons,num_iters,num_eps,n,g)
    beta,h,init_pos = td_room_policy_iter(W,phi,epsilons,num_iters,num_eps,n,g)   

    v = np.dot(phi,beta)
    I = np.zeros((n,1)); I[init_pos] = 1 
    G = np.zeros((n,1)); G[g] = 1
    spectral.plot_functions_on_room(wall_size,np.hstack((G,h,v)))
    #spectral.plot_functions_on_room(wall_size,v)

def main():
    policy_iteration()


if __name__ == '__main__':
    main()
            
       




