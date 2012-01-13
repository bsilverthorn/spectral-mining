import numpy
import scipy.sparse
import sklearn.neighbors
import specmine
import copy

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

def test_gen_templates(num_tests=10,m=2,n=2,size=9):
    feat = TemplateFeature(numpy.zeros((m,n)),(0,0))
    features = feat.gen_features()

    try:
        for i in xrange(num_tests):
            ft = TemplateFeature(numpy.array(numpy.round( \
                2*numpy.random.random((m,n))-1), dtype=int), \
                (int(round((size-m)*numpy.random.random())), \
                int(round((size-n)*numpy.random.random()))))
            assert ft in features

    except AssertionError:
        print ft._string + 'is not in the templates list'
    
class TemplateFeature(object):

    def __init__(self,grid,position):
        self.grid = grid
        self.position = position
        self._string = str(grid.flatten())


    def __hash__(self,size=9):
        boards = []
        m,n = self.grid.shape
        board = numpy.zeros((size,size))
        board[self.position[0]:self.position[0]+m, \
            self.position[1]:self.position[1]+n] = self.grid
        boards.append(board)
        

        return hash(self._string)

    def __eq__(self,other):
        return self._string == other._string

    def gen_templates(self, pref_list=[], templates = set()):
        ''' generate the full list of templates of the same size as
        the current feature'''

        n,m = self.grid.shape
        if len(pref_list) == n*m:
            templates.add(tuple(pref_list))
        else:
            for s in [-1, 0, 1]:
                new_list = copy.copy(pref_list)
                new_list.append(s)
                self.gen_templates(new_list,templates)

            return list(templates)

    def gen_features(self,size=9):
        '''generate a list of all nxm templates in all positions on the board, 
        (taking into account symmetries) '''
        templates = self.gen_templates()
        features = set()
        n,m = self.grid.shape
        for i in xrange(size-m+1):
            for j in xrange(size-n+1):
                for temp in templates:
                    features.add(TemplateFeature( \
                        numpy.reshape(numpy.array(temp),(n,m)),(i,j)))

        return list(features)
            
class TemplateFeatureMap(object):
    
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

class InterpolationFeatureMapRaw(object):
    """Map states to features via nearest-neighbor regression."""

    def __init__(self, basis, ball_tree, affinity_map, k = 8):
        self.basis = basis
        self.ball_tree = ball_tree
        self.affinity_map = affinity_map
        self.k = k
        self._D = basis.shape[1]

    def __getitem__(self, state):
        affinity_vector = self.affinity_map(state)

        (d, i) = self.ball_tree.query(affinity_vector, k = self.k, return_distance = True)

        # simple nearest neighbor averaging
        return numpy.dot(d / numpy.sum(d), self.basis[i, :])[0, 0, :]

class ClusteredFeatureMap(object):
    """Map states to features modulo clustering."""

    def __init__(self, affinity_map, clustering, maps):
        self._affinity_map = affinity_map
        self._clustering = clustering
        self._maps = maps
        self._D = sum(m._D for m in maps)

        self._indices = numpy.zeros((len(maps), self._D), numpy.uint8)

        dd = 0

        for (k, map_) in enumerate(maps):
            self._indices[k, dd:dd + map_._D] = 1

            dd += map_._D

        assert len(maps) == len(clustering.cluster_centers_)

    def __getitem__(self, state):
        avector = self._affinity_map(state).astype(float)
        (k,) = self._clustering.predict(avector[None, :])

        features = numpy.zeros(self._D)
        features[self._indices[k]] = self._maps[k][state]

        return features

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

def affinity_graph(vectors_ND, neighbors, sigma = 1e16, get_tree = False):
    """Build the k-NN affinity graph from state feature vectors."""

    G = neighbors
    (N, D) = vectors_ND.shape

    logger.info("building balltree in %i-dimensional affinity space", D)

    tree = sklearn.neighbors.BallTree(vectors_ND)

    logger.info("retrieving %i nearest neighbors", G)

    (neighbor_distances_NG, neighbor_indices_NG) = tree.query(vectors_ND, k = G)

    logger.info("constructing the affinity graph over %i vertices", N)

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
    coo_affinities = numpy.exp(-coo_distances**2 / (2.0 * sigma))

    adjacency = scipy.sparse.coo_matrix((coo_affinities, (coo_is + coo_js, coo_js + coo_is)))

    logger.info("affinity graph has %i unique edges", coo_affinities.shape[0])

    if get_tree:
        return (adjacency, tree)
    else:
        return adjacency

    ## cluster states
    #logger.info("aliasing states with spectral clustering")

    #clustering = sklearn.cluster.SpectralClustering(mode = "arpack")

    #clustering.fit(affinity_lil_NN.tocsr())

    ## cluster states directly
    #K = clusters
    #clustering = sklearn.cluster.KMeans(k = K)

    #clustering.fit(features_ND)


if __name__ == "__main__":
    test_gen_templates()
