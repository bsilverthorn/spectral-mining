import numpy as np
import scipy.sparse
import scipy.sparse.linalg
import sklearn.cluster
import specmine

logger = specmine.get_logger(__name__)

try:
    import pyamg
except ImportError:
    pass
    # XXX
    #logger.warning("failed to import pyamg; adaptive multigrid disabled")

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

def laplacian_basis(W, k, largest = False, method = "arpack"):
    """Build laplacian basis matrix with k bases from weighted adjacency matrix W."""

    logger.info(
        "solving for %i %s eigenvector(s) of the Laplacian (using %s)",
        k,
        "largest" if largest else "smallest",
        method,
        )

    L = laplacian_operator(W) 

    assert isinstance(L, scipy.sparse.csr_matrix)
    assert k > 0
    assert k < L.shape[0]

    if method == "amg":
        solver = pyamg.smoothed_aggregation_solver(L)
        pre = solver.aspreconditioner()
        initial = scipy.rand(L.shape[0], k)

        (evals, basis) = scipy.sparse.linalg.lobpcg(L, initial, M = pre, tol = 1e-10, largest = largest)
        logger.info('amg eigen values: '+str(evals))
    elif method == "arpack":
        logger.info('using arpack')
        if largest:
            which = "LM"
        else:
            which = "SM"

        if hasattr(scipy.sparse.linalg, "eigsh"): # check scipy version
            which = "LM" # use sigma=0 and ask for the large eigenvalues (shift trick, see arpack doc)
            (evals, basis) = scipy.sparse.linalg.eigsh(L, k, which = which, tol=1e-10, sigma = 0, maxiter=15000)
            try:
                for i in xrange(len(basis)):
                    b = basis[i]
                    print 'basis vector shape: ', b.shape
                    residual = np.linalg.norm((np.dot(L,b)).todense()-evals[i]*b)
                    perp_test = np.linalg.norm(np.dot(basis[i],basis[1]))
                    logger.info('eigenvalue residual: %f',residual)
                    logger.info('dot of %ith eigenvector with first: %f',i,perp_test)

            except:
                print 'error in eigensolver test code'

            logger.info('arpack eigen values: '+str(evals))
        else: 
            (evals, basis) = scipy.sparse.linalg.eigen_symmetric(L, k, which = which)
            logger.info('arpack (old) eigen values: '+str(evals))
    elif method == "dense":
        (evals, full_basis) = np.linalg.eigh(L.todense())

        basis = full_basis[:, :k]
    else:
        raise ValueError("unrecognized eigenvector method name")

    assert basis.shape[1] == k

    return basis

def clustered_laplacian_basis(W, num_clusters, num_evs, affinity_vectors, method = "arpack"):
    ''' cluster the adjacency graph before solving for laplacian eigenvectors on the subgraph 
        reorders the indices of the graph, intended for use with interpolation feature map'''

    # adjacency graph must have integer values
    A = scipy.sparse.csr_matrix(W); A.data[:] = 1
    A.data = np.array(A.data,dtype=int)

    assert isinstance(A, scipy.sparse.csr_matrix)
    assert num_clusters < A.shape[0]

    clustering = specmine.graclus.cluster(A,num_clusters)

    B = np.zeros((A.shape[0],num_clusters*num_evs))
    lef = 0; bot = 0
    reordering = np.array([],dtype=int)
    for c in xrange(num_clusters):

        order = (clustering == c).nonzero()[0]
        reordering = np.hstack((reordering,order))

        #cluster_indxs = clustering == c
        a = A[order,:]
        a = a[:,order]
        b = laplacian_basis(a, num_evs, method = method)
        B[bot:bot+b.shape[0],lef:lef+b.shape[1]] = b
        bot = bot + b.shape[0]
        lef = lef + b.shape[1]

    reord_aff_vec = affinity_vectors[reordering,:]
    #B = scipy.sparse.csr_matrix(B) 
    assert bot == W.shape[0]
    assert lef == num_clusters*num_evs
    assert affinity_vectors.shape == reord_aff_vec.shape
    assert B.shape[0] == reord_aff_vec.shape[0]

    return B, reord_aff_vec

def diffusion_basis(W,k,J=8,lam=2.5,p=1,eps_scal=10**-3):
    ''' build diffusion wavelet basis matrix with k bases from weighted adjacency matrix W '''
    T = specmine.spectral.diffusion_operator(W)
    n = T.shape[0] # number of states in complete space
    phi_dict,psi_dict = dw_tree(T,J,lam,p,eps_scal)

    return expand_wavelets(phi_dict, psi_dict, k, n)



def clustered_basis_from_affinity(avectors_ND, B, clusters = None, neighbors = 8):
    avectors_ND = np.asarray(avectors_ND, dtype = float)

    (N, D) = avectors_ND.shape

    if clusters is None:
        clusters = int(round(N / 10000.0))

    K = clusters

    logger.info("finding %i clusters over %i points in %i-dimensional affinity space", K, N, D)

    #k_means = sklearn.cluster.KMeans(k = K)
    k_means = sklearn.cluster.MiniBatchKMeans(k = K)

    k_means.fit(avectors_ND)

    logger.info("computing %i basis vectors for each cluster", B)

    blocks = []

    for k in xrange(K):
        avectors_CD = avectors_ND[k_means.labels_ == k]
        (C, _) = avectors_CD.shape

        logger.info("cluster %i (of %i) contains %i points", k + 1, K, C)

        (affinity_CC, tree) = specmine.discovery.affinity_graph(avectors_CD, neighbors, sigma = 1e16, get_tree = True)
        basis_CB = specmine.spectral.laplacian_basis(affinity_CC, min(B, C), method = "arpack")

        blocks.append((basis_CB, tree))

    return (k_means, blocks)

