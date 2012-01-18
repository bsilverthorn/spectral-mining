import random
import numpy
import scipy.sparse
import sklearn.neighbors
import specmine
import copy
import specmine.go.not_go_loops

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

def test_gen_templates(num_tests=10,m=3,n=3,size=9):
    feat = TemplateFeature(numpy.zeros((m,n)),(0,0))
    features, templates, grids = feat.gen_features()

    try:
        for i in xrange(num_tests):
            ft = TemplateFeature(numpy.array(numpy.round( \
                2*numpy.random.random((m,n))-1), dtype=numpy.int8), \
                (int(round((size-m)*numpy.random.random())), \
                int(round((size-n)*numpy.random.random()))))
            assert ft in features

    except AssertionError as e:
        print e.message
        print ft._string + ' at ' + str(ft.position) + ' is not in the features list'

def test_gen_applications(m=2,n=2,size=9):
    ''' tests generating applications for go_loops '''
    feat = TemplateFeature(numpy.zeros((m,n)),(0,0))
    features, templates = feat.gen_features()
    grids = [temp.grid for temp in templates]
    print 'number of features: ', len(features)
    print 'number of templates: ', len(grids)
    assert len(templates) == 3**(n*m)
    assert templates.index(templates[0]) == 0
    assert features.index(features[10]) == 10

    applications = []
    num_features = len(features)
    for i in xrange(num_features):
        feat = features[i]
        for app in feat.applications:
            pos = app[0]
            grid = app[1]
            try:
                applications.append([pos[0],pos[1],templates.index(Template(grid)),features.index(feat)])
            except ValueError as e:
                print e.message
                print pos
                print grid
                print feat
                print templates.index(grid)
                print features.index(feat)

    # change to unsigned?
    board = numpy.array(numpy.round(numpy.random.random((size,size))*2-1),dtype=numpy.int8)
    grids = numpy.array(grids, dtype=numpy.int8)
    applications = numpy.array(applications, dtype=numpy.int32)
    assert grids.shape[1:] == (m,n)
    print applications[0]
    print grids[0]
    indices, count = specmine.go.not_go_loops.applyTemplates(applications,grids,board,num_features)
    #indices, count = specmine.go.applyTemplates(applications,grids,board,num_features)
    
    print 'nonzero: ', indices
    print 'count: ', count

def test_template_feat_map(m=2,n=2,size=9,num_runs=10):
    #TODO test
    feat_map = TemplateFeatureMap(m,n,50)
    
    for i in xrange(num_runs):
        board = numpy.array(numpy.round(numpy.random.random((size,size))*2-1),dtype=numpy.int8)
        feat_vec = feat_map[specmine.go.GameState([],specmine.go.BoardState(board))]
        print 'number of active features: ', numpy.sum(feat_vec)
    

class Template(object):

    def __init__(self,grid):
        self.grid = grid
        self._string = str(grid.flatten())

    def __hash__(self):
        return hash(self._string)

    def __eq__(self,other):
        return self._string == other._string

class TemplateFeature(object):

    def __init__(self,grid,position, size=9):
        self.grid = grid
        self.position = position
        self.size = size
        m,n = self.grid.shape
        self.board = numpy.zeros((size,size)) # board containing only template
        self.board[self.position[0]:self.position[0]+m, \
            self.position[1]:self.position[1]+n] = self.grid
        self.board = numpy.array(self.board,dtype=numpy.int8)
        self._string = str(self.board.flatten())

        # find all symmetrically equivalent features
        self.boards = set() # list of symmetric/equivalent board strings
        offsets = [[0,0],[m-1,0],[m-1,n-1],[0,n-1],[0,n-1],[m-1,0],[m-1,n-1],[0,0]]
        self._num_apps = 0 # number of applications of the template
        self.applications = []
        for i in xrange(4):
            self.boards.add(str(numpy.rot90(self.board,i).flatten()))
            
            # if not a duplicate application, add to lists
            if len(self.boards) > self._num_apps: self.add_application(offsets,i) 

        self.boards.add(str(numpy.fliplr(self.board).flatten()))
        if len(self.boards) > self._num_apps: self.add_application(offsets,4) 
        self.boards.add(str(numpy.flipud(self.board).flatten()))
        if len(self.boards) > self._num_apps: self.add_application(offsets,5) 
        rot_board = numpy.rot90(self.board,1)
        self.boards.add(str(numpy.fliplr(rot_board).flatten()))
        if len(self.boards) > self._num_apps: self.add_application(offsets,6) 
        self.boards.add(str(numpy.flipud(rot_board).flatten()))
        if len(self.boards) > self._num_apps: self.add_application(offsets,7) 


    def add_application(self,offsets,i,size=9):
        self._num_apps += 1
        rot_brd = numpy.zeros((size,size))
        rot_brd[self.position[0],self.position[1]] = 1


        if i < 4:
            rot_template = numpy.rot90(self.grid,i) # rotated template
            pp = numpy.nonzero(numpy.rot90(rot_brd,i))
        elif i == 4:
            rot_template = numpy.fliplr(self.grid)
            pp = numpy.nonzero(numpy.fliplr(rot_brd))
        elif i == 5:
            rot_template = numpy.flipud(self.grid)
            pp = numpy.nonzero(numpy.flipud(rot_brd))
        elif i == 6:
            rot_template = numpy.fliplr(numpy.rot90(self.grid,1))
            pp = numpy.nonzero(numpy.fliplr(numpy.rot90(rot_brd)))
        elif i == 7:
            rot_template = numpy.flipud(numpy.rot90(self.grid,1))
            pp = numpy.nonzero(numpy.flipud(numpy.rot90(rot_brd)))

        pp = numpy.array(pp,dtype=numpy.int8).flatten() - numpy.array(offsets[i],dtype=numpy.int8) # new top left position
        self.applications.append([pp,rot_template]) 
        assert (-1 < pp[0] & pp[0] < size) & (-1 < pp[1] & pp[1] < size)

    def __hash__(self):
        
        hashes = map(hash,self.boards)
        return max(hashes)

    def __eq__(self,other):

        # test - TODO remove
        if self._string in other.boards:
            #print 'string: ', self._string 
            #print 'boards: ', self.boards
            assert self._string in self.boards
        else:
            #print 'string: ', other._string 
            #print 'boards: ', self.boards
            assert not other._string in self.boards

        return self._string in other.boards

    def gen_templates(self, pref_list=[], templates = set()):
        ''' generate the full list of templates of the same size as
        the current feature without duplicates'''

        n,m = self.grid.shape
        if len(pref_list) == n*m:
            grid = numpy.reshape(numpy.array(pref_list,dtype=numpy.int8),(n,m))
            templates.add(Template(grid))
        else:
            for s in [-1, 0, 1]:
                new_list = copy.copy(pref_list)
                new_list.append(s)
                self.gen_templates(new_list,templates)

            return list(templates)

    def gen_features(self,size=9):
        '''generate a list of all nxm templates in all positions on the board, 
        (taking into account symmetries) '''
        templates = self.gen_templates() # generate templates without duplicates
        features = set()
        n,m = self.grid.shape
        for i in xrange(size-m+1):
            for j in xrange(size-n+1):
                for temp in templates:
                    features.add(TemplateFeature(temp.grid,(i,j)))

        random.shuffle(list(features))
        return features, templates
            
class TemplateFeatureMap(object):

    def __init__(self, m, n, B,size=9):
        feat = TemplateFeature(numpy.zeros((m,n)),(0,0))
        self.features, self.templates = feat.gen_features()
        self.grids = [temp.grid for temp in self.templates]
        self.B = B # number of active basis (other than flattened grid
        self.applications = []
        print type(self.features)
        print self.features
        self.num_features = len(self.features) # total number of possible features
        for i in xrange(min(B,self.num_features)):
            feat = self.features[i]
            for app in feat.applications:
                pos = app[0]
                grid = app[1]
                self.applications.append([pos[0],pos[1],self.templates.index(Template(grid)),self.features.index(feat)])

        self.grids = numpy.array(self.grids, dtype=numpy.int8)
        self.applications = numpy.array(self.applications, dtype=numpy.int32)

        # TODO - save/load feature map

    def __getitem__(self, state):
        grid = state.board.grid
        
        if self.B > 0:
            indices, count = specmine.go.not_go_loops.applyTemplates( self.applications, self.grids, grid, self.num_features)
            #indices, count = specmine.go.applyTemplates(self.applications,self.grids,board,num_features)

        feat_vec = numpy.zeros(B)
        if B > 0:
            feat_vec[indices] = 1
            # feat_vec[indices] = counts # weighted by number of occurances
        
        return numpy.hstack((1,grid.flatten(),feat_vec)) # add constant and float board vector


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
    test_template_feat_map()
