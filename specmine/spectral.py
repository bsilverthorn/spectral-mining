import numpy as np
import scipy.sparse
import scipy.sparse.linalg
import specmine

logger = specmine.get_logger(__name__)

def laplacian_operator(W):
    n = W.shape[0]
    
    if not scipy.sparse.issparse(W):
        W = scipy.sparse.csr_matrix(W)
    W_row_sum = W.sum(1).T
    D = scipy.sparse.spdiags(W_row_sum,0,n,n)

    D_invsqrt = scipy.sparse.spdiags(1./np.sqrt(W_row_sum),0,n,n)
    return D_invsqrt*(D-W)*D_invsqrt

# funky behavior with random walk laplacian
#    D_inv = scipy.sparse.spdiags(1./(W_row_sum),0,n,n)
#    assert (D_inv*(D-W) - D_inv.dot(D-W)).sum() < 10**-4
#    return scipy.sparse.eye(n,n)-D_inv*W #D_inv*(D-W)

def diffusion_operator(W):
    D = np.diag(np.sum(W,1))
    D_invsqrt = np.sqrt(np.linalg.inv(D))

    return np.dot(np.dot(D_invsqrt,W),D_invsqrt)

def dw_tree(T,J,lam,p,eps_scal):
    phi_dict = {} # scaling functions
    psi_dict = {} # wavelet functions
    print 'building diffusion wavelet tree'
    for j in range(J):
        print 'level j = ', j
        # heuristic for setting epsilon, from Bo Liu paper:
        eps = np.sum(np.sum(T))/len(T[T>0])*eps_scal
        print 'epsilon: ', eps
        phi = sparseQR(T,eps,lam,p)
        psi = sparseQR(np.eye(T.shape[0])-np.dot(phi,phi.T),eps,lam,p)
        print 'size of scaling function basis: ', phi.shape
        if psi != None:
            print 'size of wavelet function basis: ', psi.shape
        else:
            print 'no wavelets this level'
        phi_dict[j] = phi
        psi_dict[j] = psi
        T = np.dot(np.dot(np.dot(phi.T,T),T),phi) # T^2 in k-space (kxk)

    return phi_dict, psi_dict

def expand_wavelets(phi_dict, psi_dict, k, n):
    J = len(phi_dict)
    phi_coarse = np.eye(n)
    psi_coarse = np.eye(n)
    for j in xrange(J):
        phi_coarse = np.dot(phi_coarse,phi_dict[j])
        # if the wavelet functions exist, add them to the left
        if psi_dict[j] != None:
            psi_coarse = np.dot(psi_coarse,phi_dict[j])
            if j == 0:
                basis = psi_coarse
            elif j < J-1: 
                basis = np.hstack((psi_coarse,basis))
            # on the last iteration, add the scaling functions
            else:
                basis = np.hstack((phi_coarse,basis))
        # if the wavelet functions don't exist, use the scaling functions
        # the scaling functions in this case span the whole space (no 
        # complementary space for the wavelets to pick up)
        else:
            if j == 0: 
                basis = phi_coarse
            else:
                basis = np.hstack((phi_coarse,basis))

    # add the constant vector and return 
    return np.hstack((np.ones((n,1))/float(n),basis[:,:k-1]))

def laplacian_basis(W, k, which = "SM", sparse = True):
    """Build laplacian basis matrix with k bases from weighted adjacency matrix W."""

    logger.info("solving for %i eigenvectors of the Laplacian", k)

    L = laplacian_operator(W) 

    if sparse:
        spL = scipy.sparse.csr_matrix(L)
        
        if hasattr(scipy, "sparse.linalg.eigsh"):
            (eig_lam, eig_vec) = scipy.sparse.linalg.eigsh(spL, k, which = which)
        else: 
            (eig_lam, eig_vec) = scipy.sparse.linalg.eigen_symmetric(spL, k, which = which)
        
    else:
        if scipy.sparse.issparse(L):
            L = L.todense()

        (eig_lam,eig_vec) = np.linalg.eigh(L)

    sort_inds = eig_lam.argsort()
    eig_lam = eig_lam[sort_inds]
    phi = eig_vec[:, sort_inds]
    phi = phi[:, :k]

    return phi

def diffusion_basis(W,k,J=8,lam=2.5,p=1,eps_scal=10**-3):
    ''' build diffusion wavelet basis matrix with k bases from weighted adjacency matrix W '''
    T = spectral.diffusion_operator(W)
    n = T.shape[0] # number of states in complete space
    phi_dict,psi_dict = dw_tree(T,J,lam,p,eps_scal)

    return expand_wavelets(phi_dict, psi_dict, k, n) 

