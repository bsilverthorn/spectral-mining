import numpy
import scipy.sparse
import sklearn.neighbors
import specmine

logger = specmine.get_logger(__name__)

class TabularFeatureMap(object):
    """Map states to features via simple table lookup."""

    def __init__(self, basis_matrix, index):
        self.basis = basis_matrix # number of states x number of features
        u, s, v = numpy.linalg.svd(basis_matrix)
        rank = numpy.sum(s > 1e-10)
        print 'rank of basis: ', rank
        print 'full rank: ', basis_matrix.shape[1]
        #assert rank == basis_matrix.shape[1]

        self.index = index

    def __getitem__(self, state):
        #print 'state: ',state
        return self.basis[self.index[state], :]

class RandomFeatureMap(TabularFeatureMap):
    """Map states to (fixed) random features."""

    def __init__(self, count, index):
        TabularFeatureMap.__init__(
            self,
            numpy.random.random((len(index), count)),
            index,
            )

class InterpolationFeatureMap(object):
    """Map states to features via nearest-neighbor regression."""

    def __init__(self, basis, affinity_vectors, affinity_map, k = 8):
        self.basis = basis
        self.affinity_vectors = affinity_vectors
        self.ball_tree = sklearn.neighbors.BallTree(affinity_vectors)
        self.affinity_map = affinity_map
        self.k = k

    def __getitem__(self, state):
        affinity_vector = self.affinity_map(state)

        (d, i) = self.ball_tree.query(affinity_vector, k = self.k, return_distance = True)

        # simple nearest neighbor averaging
        return numpy.dot(d / numpy.sum(d), self.basis[i, :])[0, 0, :]

def adjacency_dict_to_matrix(adict):
    """
    Create a symmetric adjacency matrix from a {state: [state]} dict.

    Return a sparse matrix in an unspecified storage format. Note that the
    value of an edge between a pair of nodes will be the number of edges that
    exist between those nodes, i.e., the matrix will consist of values in {0,
    1, 2}.
    """

    index = dict(zip(adict, xrange(len(adict))))
    coo_is = []
    coo_js = []

    for (node, neighbors) in adict.iteritems():
        n = index[node]

        for neighbor in neighbors:
            m = index[neighbor]

            coo_is.append(n)
            coo_js.append(m)

    coo_values = numpy.ones(len(coo_is) * 2, int)

    return (
        scipy.sparse.coo_matrix((coo_values, (coo_is + coo_js, coo_js + coo_is))),
        index,
        )

def adjacency_matrix_to_dict(amatrix, rindex = None, make_directed = True):
    """Create an affinity dict from a sparse adjacency matrix."""

    (N, _) = amatrix.shape
    adict = {}

    if rindex is None:
        rindex = numpy.arange(N)

    for n in xrange(N):
        (_, nonzero) = amatrix.getrow(n).nonzero()

        if make_directed:
            neighbors = [rindex[m] for m in nonzero if m > n]
        else:
            neighbors = [rindex[m] for m in nonzero if m != n]

        adict[rindex[n]] = neighbors

    return adict

def affinity_graph(vectors_ND, neighbors):
    """Build the k-NN affinity graph from state feature vectors."""

    G = neighbors
    (N, D) = vectors_ND.shape

    logger.info("building balltree in affinity space")

    tree = sklearn.neighbors.BallTree(vectors_ND)

    logger.info("retrieving nearest neighbors")

    (neighbor_distances_NG, neighbor_indices_NG) = tree.query(vectors_ND, k = G)

    logger.info("constructing the affinity graph")

    coo_is = []
    coo_js = []
    coo_distances = []

    for n in xrange(N):
        for g in xrange(G):
            m = neighbor_indices_NG[n, g]

            coo_is.append(n)
            coo_js.append(m)
            coo_distances.append(neighbor_distances_NG[n, g])

    coo_distances = numpy.array(2 * coo_distances)
    coo_affinities = numpy.exp(-coo_distances**2 / 2.0)

    adjacency = scipy.sparse.coo_matrix((coo_affinities, (coo_is + coo_js, coo_js + coo_is)))

    logger.info("done creating affinity graph adjacency matrix")

    return adjacency

    ## cluster states
    #logger.info("aliasing states with spectral clustering")

    #clustering = sklearn.cluster.SpectralClustering(mode = "arpack")

    #clustering.fit(affinity_lil_NN.tocsr())

    ## cluster states directly
    #K = clusters
    #clustering = sklearn.cluster.KMeans(k = K)

    #clustering.fit(features_ND)

