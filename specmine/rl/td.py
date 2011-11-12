def td_episode(S, R, phi, beta = None, lam=0.9, gamma=1, alpha = 0.001):

    k = phi.shape[1]
    if beta == None:
        beta = np.zeros(k)
    z = phi[S[0],:]   
    for t in xrange(len(R)):
        curr_phi = phi[S[t],:] 
        z_old = z
        if t == len(R)-1:
            # terminal state is defined as having value zero
            delta = z*(R[t]-np.dot(curr_phi,beta)) 

        else:
            delta = z*(R[t]+np.dot((gamma*phi[S[t+1],:]-curr_phi),beta))
            z = gamma*lam*z+phi[S[t+1],:]

        beta += delta*alpha/np.dot(z_old,curr_phi)

    return beta

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

