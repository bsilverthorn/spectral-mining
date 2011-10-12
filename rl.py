from random import choice
import numpy as np
import scipy.sparse
import scipy.sparse.linalg
import spectral

# TODO add gamma
def lstd_episode(S, R, phi, lam=0.9, A=None, b=None):

    k = phi.shape[1] # number of features
    if A == None:
        A = np.zeros((k,k))
    if b == None:
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

def td_episode(S, R, phi, lam=0.9, gamma=1, beta = None, alpha = 0.001):
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


def vi_episode(W, S, R, phi, lam=0, gamma=1, beta=None, alpha=0.01):
    ''' not sure if its legal/accepted to use lambda (eligibility traces) here
    with value iteration; makes it policy dependent.'''
    k = phi.shape[1]
    if beta == None:
        beta = np.zeros((k,1))
    z = phi[S[0],:]
    v = np.dot(phi,beta)

    for t in xrange(len(R)):
        # TODO add max over (simulated) rewards also - currently all rewards eql
        neighbors = np.nonzero(W[:,S[t]])[0]
        max_value = np.max(v[neighbors])
        best = v[:, 0][neighbors] == max_value
        best_next_state = choice(neighbors[best]) 

        curr_phi = phi[S[t],:] 
        delta = z[:,None]*(R[t]+np.dot((gamma*phi[best_next_state,:]-curr_phi)[None,:],beta))
        z = gamma*lam*z+phi[S[t+1],:]
        beta += delta*alpha/np.linalg.norm(curr_phi,1)

        # update value function after every transition or every episode?
        # comment below out for the latter
        v = np.dot(phi,beta)
    
    return beta


def room_walk(W,pos,v,g,eps=0.1,stop_prob=0.005):
    
    S = [pos]
    R = []
    t = 0
    # stop when goal is reached or randomly with prob stop_prob
    while (pos != g) & (np.random.rand() > stop_prob):
        # choose next state
        if np.random.rand() <= eps:
            pos = choice(np.nonzero(W[:,pos])[0]) # random step
            #print 'pos - random', pos
        else:
            neighbors = np.nonzero(W[:,pos])[0]
            max_value = np.max(v[neighbors])
            best = v[:, 0][neighbors] == max_value
            pos = choice(neighbors[best]) # greedy step
            #print 'neighbors: ', neighbors
            #print 'pos - greedy', pos

        R.append(-0.1)
        S.append(pos)
        t += 1

    return S, R, t

def heat_map(S,n):
    
    h = np.zeros((n,1))
    for s in S:
        h[s] += 1

    return h # /float(len(S))

def tworoom_PI(W,phi,epsilons,num_iters,num_eps,n,g):
    
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
                _A,_b = lstd_episode(S,R,phi)
                A += _A; b += _b; t += _t

        print 'average time steps: ', t/float(ep)
        beta = lstd_solve(A,b)
        v = np.dot(phi,beta)
    
    return beta,h, init_pos 

def td_room_policy_iter(W,phi,epsilons,num_iters,num_eps,n,g):
    k = phi.shape[1]
    weights = np.zeros((k,1))
    v = np.dot(phi,weights) # set value according to current weights, features

    #init_pos = n-1
    for i in xrange(num_iters):
        print "iteration: ", i
        print 'epsilon: ' , epsilons[i] 
        # collect num_eps episodes of data with current policy
        h = np.zeros((n,1)) 
        t = 0
        ep = 0
        #while ep < num_eps:
        for e in xrange(num_eps):
            init_pos = np.round(np.random.random()*(n-1)) # start from random position
            S,R,_t = room_walk(W,init_pos,v,g,epsilons[i]) # collect state transitions with eps-greedy walk
            if S[-1] == g:
                print 'made to goal'
             #   print ep
             #   ep += 1
            h += heat_map(S,n)  
            t += _t              
            weights = td_episode(S,R,phi,beta = weights)
            #weights = q_lambda(W,S,R,phi,beta = weights)

        print 'average time steps: ', t/float(num_eps)
        v = np.dot(phi,weights)  

    return weights, h, init_pos

def tworoom_VI(W, phi, g, num_pols=1, num_eps=500, epsilons=np.ones(500)):

    (n,k) = phi.shape
    weights = np.zeros((k,1))
    v = np.dot(phi,weights) # set value according to current weights, features

    for i in xrange(num_pols):
        print "policy: ", i
        print 'epsilon: ' , epsilons[i] 
        # collect num_eps episodes of data with current policy
        h = np.zeros((n,1)) 
        t = 0
        for e in xrange(num_eps):
            init_pos = np.round(np.random.random()*(n-1)) 
            # collect state transitions with eps-greedy walk
            S,R,_t = room_walk(W,init_pos,v,g,epsilons[i])
            S.reverse()
            R.reverse()
            #if S[-1] == g:
            #    print 'made it to goal'
            h += heat_map(S,n)  
            t += _t              
            weights = vi_episode(W, S, R, phi, beta=weights)

        print 'average time steps: ', t/float(num_eps)
        v = np.dot(phi,weights)  

    return weights, h



def build_rep_laplacian(W, k=100, sparse=False):
    
    L = spectral.laplacian_operator(W) 
    print "solving for the eigenvectors of the laplacian"
    if sparse:
        spL = scipy.sparse.csr_matrix(L)
        (eig_lam, eig_vec) = scipy.sparse.linalg.eigen_symmetric(spL, k, which = "SM")
    else:
        (eig_lam,eig_vec) = np.linalg.eigh(L)

    sort_inds = eig_lam.argsort()
    eig_lam = eig_lam[sort_inds]
    phi = eig_vec[:,sort_inds]
    phi = phi[:,:k]
    
    return phi

def build_rep_diffusion(W,k = 100):
    print "building diffusion operator and basis"
    T = spectral.diffusion_operator(W)
    return spectral.build_diffusion_basis(T,k)



def main(wall_size=20, k=30, use_VI=True, rand_walk=True, use_laplacian=True):
    print "building adjacency matrix"
    W = spectral.room_adjacency(wall_size, self_loops=False) 
    # use adjacency matrix with self-loops to form basis functions
    W_smooth = spectral.room_adjacency(wall_size, self_loops=True) 

    n = W.shape[0] # number of states
    if use_laplacian:
        phi = build_rep_laplacian(W_smooth, k)
    else: # use diffusion wavelets
        phi = build_rep_diffusion(W_smooth,k)

    g = 2*wall_size + 2 # goal is a position near the corner, not on a wall

    if rand_walk:
        num_pols = 1 # only one policy being evaluated/used
        num_eps = 5000 # number of episodes 
        epsilons = np.ones(num_eps)
        print 'using random walk, one policy ', num_eps,' episodes per policy'
    else:
        num_pols = 10 
        num_eps = 300 #number of episodes per policy eval/sampling according to pol
        epsilons = 0.1 * np.ones(num_eps); epsilons[0:4] = 1
        #epsilons = 1*np.exp(-np.array(range(num_iters))/(num_iters/2.)) 
        print 'using ',num_pols,' policies with ',num_eps,' episodes per policy'
        
    if use_VI: 
        print 'performing value iteration'
        beta, h = tworoom_VI(W, phi, g, num_pols, num_eps, epsilons)
    else: 
        print 'performing policy iteration'
        #beta,h = lstd_room_policy_iter(W,phi,epsilons,num_pols,num_eps,g) # TODO merge PI
        beta,h = td_room_policy_iter(W,phi,epsilons,num_pols,num_eps,g)   

    v = np.dot(phi,beta) # final value function
    G = np.zeros((n,1)); G[g] = 1 # goal position
    spectral.plot_functions_on_room(wall_size,np.hstack((G,h,v)))


if __name__ == '__main__':
    main()
            
       





