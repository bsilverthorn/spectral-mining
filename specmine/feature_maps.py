import sys
import cPickle as pickle
import random
import numpy
import scipy.sparse
import sklearn.neighbors
import specmine
import copy
# import specmine.go.not_go_loops
import specmine.go.go_loops

logger = specmine.get_logger(__name__)

def flat_affinity_map(state):
    ''' takes a Go (Game- or Board-) State and returns the affinity map representation -- currently
    the flattened board'''
    if state.__class__ == specmine.go.BoardState:
        return state.grid.flatten()
    elif state.__class__ == specmine.go.GameState:
        return state.board.grid.flatten

class TabularFeatureMap(object):
    """Map states to features via simple table lookup."""

    def __init__(self, basis_matrix, index):
        self.basis = basis_matrix # number of states x number of features
        self.index = index

    def __getitem__(self, state):
        #print 'state: ',state
        if self.basis is not None:
            return self.basis[self.index[state]]
        else:
            return numpy.array([])

class RandomFeatureMap(TabularFeatureMap):
    """Map states to (fixed) random features."""

    def __init__(self, count, index):
        TabularFeatureMap.__init__(
            self,
            numpy.random.random((len(index), count)),
            index,
            )

class InterpolationFeatureMap(object):
    """Map states to features via nearest-neighbor regression. Must give either
    a ball tree or the raw affinity vectors (not both)"""

    def __init__(self, basis, affinity_map, ball_tree = None, affinity_vectors=None, k = 8, sigma_sq = -1):
        self.basis = basis
        self.sigma = sigma_sq # parameter for neighbor weighting
       
        if ball_tree is None:
            self.affinity_vectors = affinity_vectors
            self.ball_tree = sklearn.neighbors.BallTree(affinity_vectors)
        else:
            assert affinity_vectors is None
            self.ball_tree = ball_tree

        self.affinity_map = affinity_map
        self.k = k
        print str.format('using {a} neighbors for interpolation',a=k)

    def __getitem__(self, state):
        affinity_vector = self.affinity_map(state)

        (d, i) = self.ball_tree.query(affinity_vector, k = self.k, return_distance = True)

        # simple nearest neighbor averaging
        if self.basis is not None:
            #logger.info('average distance: %f',numpy.mean(d))
            #logger.info('distance variance: %f',numpy.var(d))
            #print 'squared distance: ', d**2
            #print 'distance : ', d
    
            if self.sigma == -1:
                weighting = numpy.ones(self.k)/float(self.k)
                return numpy.dot(weighting, self.basis[i, :]).flatten()
            else:
                weighting = numpy.exp(-d**2/self.sigma_sq)
                weighting = weighting/sum(weighting)
                return numpy.dot(weighting, self.basis[i, :]).flatten()
            
        else:
            return numpy.array([])

    def gen_map(self,B):
        '''generate a copied feature map with fewer features'''
        interp_map = copy.deepcopy(self)
        interp_map.basis = interp_map.basis[:,B]
        return interp_map


class ClusteredFeatureMap(object):
    """Map states to features modulo clustering."""
    # TODO - add board vector and constant features?
    # TODO - not maintained
    
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
        

class TemplateFeatureMap(object):
    ''' computes mxn template features vector given a board or game state. The
    template features are rotation/reflection invariant and are position 
    dependent on the board. The feature vector also always includes a constant 
    and includes flattened board if append_board=True'''

    def __init__(self, m, n, NF = numpy.inf, size=9, weighted = False):
        
        self._weighted = weighted

        # generate the feature map from all possible mxn templates
        logger.info("generating %i by %i template features",m,n)

        feat = TemplateFeature(numpy.zeros((m,n)),(0,0))
        self.features, templates = feat.gen_features(max_num = NF)

        assert (len(self.features) == NF) | (NF == numpy.inf)

        self.NF = len(self.features) # number of features
        self.grids = [temp.grid for temp in templates]
        self.applications = []
        
        logger.info("number of %iby%i features: %i",m,n, self.NF)
            
        # from the templates and features list, construct the applications for use in go_loops
        for i in xrange(len(self.features)):
            feat = self.features[i]
    
            for app in feat.applications:
                pos = app[0]
                grid = app[1]
                assert grid.shape == (m,n)
                self.applications.append([pos[0], pos[1], templates.index(Template(grid)),self.features.index(feat)])
            
        # TODO for non-square feature debugging, can be removed 
        for g in self.grids:
            try:
                assert g.shape == (m,n)
            except AssertionError:
                print m,n
                print 'not m by n grid: ', g.shape
                print g
                sys.exit()

        self.grids = numpy.array(self.grids, dtype=numpy.int8)
        self.applications = numpy.array(self.applications, dtype=numpy.int32)


    def __getitem__(self, state):
        ''' computes the feature vector given a board or game state '''
        
        if state.__class__ == specmine.go.BoardState:
            board = state.grid
        elif state.__class__ == specmine.go.GameState:
            board = state.board.grid
        
        if self.NF > 0:
            indices, counts = specmine.go.applyTemplates(self.applications,self.grids,board,self.NF)
            feat_vec = numpy.zeros(self.NF)

            if self._weighted:
                feat_vec[indices] = counts # weighted by number of occurances
            else:
                feat_vec[indices] = 1
        else:
            feat_vec = []
        
        return numpy.array(feat_vec)

    def gen_map(self, NF):
        ''' template feature maps can generate as big or smaller tfm objects using precomputed values '''
        if NF > 0:
            feat_nums = self.applications[:,3]
            applications = self.applications[feat_nums<NF,:]
            return RawTemplateFeatureMap(self.features, self.grids, applications, NF)
        else:
            m,n = self.grids[0].shape
            return TemplateFeatureMap(m,n,NF)
            
        

class RawTemplateFeatureMap(TemplateFeatureMap):
    
    def __init__(self, features, grids, applications, NF, weighted = False):
        self.features = features
        self.applications = applications
        self.grids = grids
        self._weighted = weighted
        self.NF = NF

class Template(object):
    ''' hashable wrapper for an mxn grid '''

    def __init__(self,grid):
        self.grid = grid
        self._string = str(grid.flatten())

    def __hash__(self):
        return hash(self._string)

    def __eq__(self,other):
        return self._string == other._string

class TemplateFeature(object):
    ''' A template feature is defined as an mxn template with a top left position
    at which to apply it on the board. hash and eq make features 
    rotationally/reflectionally invariant.'''

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
        
        # find the new top left position after rotation/reflection
        pp = numpy.array(pp,dtype=numpy.int8).flatten() - numpy.array(offsets[i],dtype=numpy.int8) 
        self.applications.append([pp,rot_template]) 
        try:
            assert (-1 < pp[0] & pp[0] < size) & (-1 < pp[1] & pp[1] < size)
        except AssertionError as e:
            print e.message
            print 'top corner out of bounds'
            print 'original board: ', rot_brd
            print 'application i = ',i
            print pp

    def __hash__(self):

        # choose the max hash of all boards as the hash
        hashes = map(hash,self.boards)
        return max(hashes)

    def __eq__(self,other):

        # test - TODO move to test
        if self._string in other.boards:
            #print 'string: ', self._string 
            #print 'boards: ', self.boards
            assert other._string in self.boards
        else:
            #print 'string: ', other._string 
            #print 'boards: ', self.boards
            assert not other._string in self.boards

        return self._string in other.boards

    def gen_templates(self, pref_list=[], templates = set()):
        ''' generate the full list of templates of the same size as
        the current feature without duplicates'''

        m,n = self.grid.shape
        if len(pref_list) == n*m:
            grid = numpy.reshape(numpy.array(pref_list,dtype=numpy.int8),(m,n))
            templates.add(Template(grid))
            assert grid.shape == (m,n)
        else:
            for s in [-1, 0, 1]:
                new_list = copy.copy(pref_list)
                new_list.append(s)
                self.gen_templates(new_list,templates)

            return list(copy.deepcopy(templates))

    def gen_features(self, max_num = numpy.inf, size=9):
        '''generate a list of all nxm templates in all positions on the board, 
        (taking into account symmetries) '''

        if max_num == 0:
            return [], []
        
        print str.format('generating {i} features',i=max_num)

        features = set()
        m,n = self.grid.shape
        templates = self.gen_templates(pref_list=[], templates = set()) # generate templates without duplicates
        for i in xrange(size-m+1):
            for j in xrange(size-n+1):
                for temp in templates:
                    features.add(TemplateFeature(temp.grid,(i,j)))
                    if len(features) >= max_num:
                        feature_list = list(features)
                        random.shuffle(feature_list)
                        return feature_list, templates

        feature_list = list(features)
        random.shuffle(feature_list)
        return feature_list, templates


def build_affinity_graph(vectors_ND, neighbors, get_tree = False):
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
    #coo_distances = []

    for n in xrange(N):
        for g in xrange(G):
            m = neighbor_indices_NG[n, g]

            coo_is.append(n)
            coo_js.append(m)
            #coo_distances.append(neighbor_distances_NG[n, g])

    #coo_distances = numpy.array(2 * coo_distances)
    #coo_affinities = numpy.exp(-coo_distances**2 / (2.0 * sigma))
    
    # just uses a weight of 1 for all edges
    coo_affinities = numpy.ones(2*len(coo_is))
    adj1 =scipy.sparse.coo_matrix((numpy.ones(len(coo_is)), (coo_is, coo_js)))
    adj2 = numpy.zeros((N,N))
    adj2[coo_is,coo_js] = 1
    adjacency = scipy.sparse.coo_matrix((coo_affinities, (coo_is + coo_js, coo_js + coo_is)))

    assert (adj2 == adj1.todense()).all()
    assert (adjacency.todense() == (adj1+adj1.T).todense()).all()
    test_adj = adjacency.todense()
    assert (test_adj.T == test_adj).all()

    if get_tree:
        return (adjacency, tree)
    else:
        return adjacency


        
