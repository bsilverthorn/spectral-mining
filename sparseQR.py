import numpy as np

def remove_rounding_error(Q, vec):
    for i in range(Q.shape[1]):
        vec = vec - np.array([np.dot(vec.T,Q[:,i])*Q[:,i]]).T
    return vec

def sparseQR(T,eps,lam,p):
    ''' returns a QR factorization of T with orthonormal elements in the columns
    of Q which have smaller support than standard QR. '''
    Q = None
    n = T.shape[0] # dimension of the col vectors
    
    Phi = T # working set of remaining col vectors
    if (Phi == np.zeros(Phi.shape)).all():
        print 'sparseQR asked to factorize zeros operator'
        return None
    
    while True: # until return condition below is met
        num_cols = Phi.shape[1]
        L2_norms = np.zeros([num_cols,1])
        for i in range(num_cols):
            L2_norms[i] = np.linalg.norm(Phi[:,i],ord=2)
            
        N = np.max(L2_norms)/lam # threshold for L2-norm of vectors to be selected
        
        vec_set_indxs = np.nonzero(L2_norms>=N)[0]
        vec_set = Phi[:,vec_set_indxs] # all vectors with L2 norm > max/lam
        num_vecs = len(vec_set_indxs) 
        Lp_norms = np.zeros([num_vecs,1])
        for i in range(num_vecs):
            Lp_norms[i] = np.linalg.norm(Phi[:,vec_set_indxs[i]],ord=p)
        
        sparse_vec_indx = vec_set_indxs[np.argmin(Lp_norms)]
        
        # if the (L2) norm of the vector selected is below the threshold, we're done
        if (L2_norms[sparse_vec_indx] < eps):
            print 'below threshold'
            print 'L2 norm of the vector not added', L2_norms[sparse_vec_indx]
            break
        
        vec = np.array([Phi[:,sparse_vec_indx]]).T
#        print 'L2 norm of vector added', np.linalg.norm(vec)
#        print 'Lp norm of vector added', np.linalg.norm(vec,ord=p)
        
        if Q == None:
            Q = vec
        else:
            # remove any components of the previously selected basis vectors from vec
            # (to compensate for rounding error)
            vec = remove_rounding_error(Q, vec)
            vec = vec/np.linalg.norm(vec) # normalize
            Q = np.hstack((Q,vec)) # add vec to ON basis
        
        # if vector just added to Q was the last in Phi, stop
        if (Phi.shape[1] == 1):
            print 'out of Phi'
            print 'norm of last vector added', L2_norms[sparse_vec_indx]
            break
        # remove the selected vector from remaining basis set, Phi
        Phi = np.hstack((Phi[:,0:sparse_vec_indx],Phi[:,sparse_vec_indx+1:]))
        # and orthogonalize the remaining vectors in Phi wrt vec
        Phi = Phi - np.tile(np.dot(vec.T,Phi),(n,1))*np.tile(vec,(1,Phi.shape[1]))
        
    #R = np.dot(Q.T,T)
    return Q #,R
