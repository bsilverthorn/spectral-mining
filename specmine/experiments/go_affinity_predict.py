import csv
import numpy
import cPickle as pickle
import condor
import specmine

logger = specmine.get_logger(__name__)

def affinity_map(state):
    ''' takes a Go State and returns the affinity map representation -- currently
    the flattened board'''
    if state.__class__ == specmine.go.BoardState:
        return state.grid.flatten()
    elif state.__class__ == specmine.go.GameState:
        return state.board.grid.flatten()


def run_template_features(m,n,B,values):

    feature_map = specmine.discovery.TemplateFeatureMap(m,n,B)
    m,n = feature_map.grids[0].shape
    (mean, variance) = specmine.science.score_features_predict(feature_map, values)
    return [str(m)+'by'+str(n)+"template", feature_map.B, None ,mean, variance]

def run_features(map_name, B, all_features_NF, affinity_vectors, index, values, interpolate = False, **kwargs):
    if interpolate:
        aff_map = kwargs['aff_map']
        feature_map = specmine.discovery.InterpolationFeatureMap(all_features_NF, affinity_vectors, aff_map)
    else:
        feature_map = specmine.discovery.TabularFeatureMap(all_features_NF, index)
    (mean, variance) = specmine.science.score_features_predict(feature_map, values)

    logger.info("with %i %s features, mean score is %.4f", B, map_name, mean)

    num_samples = feature_map.basis.shape[0]
    return [map_name, B, num_samples, mean, variance]

def run_laplacian_features(map_name, B, vectors_ND, affinity_NN, index, values, interpolate = False, **kwargs):
    if B > 0:
        basis_NB = specmine.spectral.laplacian_basis(affinity_NN, B, method = "arpack")
        all_features_NF = numpy.hstack([vectors_ND, basis_NB])
    else:
        all_features_NF = vectors_ND

    return run_features(map_name, B, all_features_NF, vectors_ND, index, values, interpolate, **kwargs)

def run_clustered_laplacian_features(map_name, evs_per_cluster, vectors_ND, affinity_NN, index, values,\
                            num_clusters, interpolate = True, **kwargs):
    if evs_per_cluster > 0:
        basis, vectors_ND = specmine.spectral.clustered_laplacian_basis(affinity_NN, num_clusters, \
                                                    evs_per_cluster, vectors_ND, method = "arpack")

        logger.info("number of laplacian features: %i", basis.shape[1])
        
        all_features_NF = numpy.hstack((vectors_ND, basis))
    else:
        all_features_NF = vectors_ND

    return run_features(map_name, evs_per_cluster, all_features_NF, vectors_ND, index, values, interpolate, **kwargs)


def run_random_features(B, vectors_ND, index, values, interpolate = False, **kwargs):
    (N, _) = vectors_ND.shape

    random_basis_NB = numpy.random.random((N, B))
    all_features_NF = numpy.hstack([vectors_ND, random_basis_NB])

    return run_features("random", B, all_features_NF, vectors_ND, index, values, interpolate, **kwargs)
    

def get_value_list(games_path,values_path):
    #games_path = specmine.util.static_path(games_path)
    #values_path = specmine.util.static_path(values_path)

    with specmine.util.openz(games_path) as games_file:
        games = pickle.load(games_file)

    with specmine.util.openz(values_path) as values_file:
        values = pickle.load(values_file)

    value_list = []

    for game in values.keys():
        try:
            vals = values[game]
            boards = list(set(map(specmine.go.BoardState, games[game].grids)))
            value_list.extend(zip(boards,vals))
        except KeyError:
            print 'game unkown for ',game

    # remove duplicates
    value_dict = dict(value_list) 
    value_keys = value_dict.keys()
    value_list = zip(value_keys,map(value_dict.__getitem__,value_keys))

    return value_list

@specmine.annotations(
    out_path = ("path to write CSV",),
    games_path = ("path to adjacency dict"),
    values_path = ("path to value function",),
    neighbors = ("number of neighbors", "option", None, int),
    workers = ("number of condor jobs", "option", None, int),
    )
def clustered_affinity_test(out_path, games_path, values_path, neighbors = 8, workers = 0, interpolate = True, off_graph = False):
    ''' value prediction using features learned from clustered graph '''
    value_list = get_value_list(games_path,values_path) 

    logger.info("number of value samples total: %i", len(value_list))
 
    def yield_jobs():
        min_samples =  20000
        max_samples = 260000
        step_samples = 60000
        cluster_size = 10000 #average
        max_test_samples = 100000
 
        shuffled_values = sorted(value_list, key = lambda _: numpy.random.rand()) 

        for samples in xrange(min_samples,max_samples,step_samples):

            num_clusters = int(round(samples/cluster_size))
    
            logger.info("number of clusters used: %i", num_clusters)
        
            # randomly sample subset of games 
            value_dict = dict(shuffled_values[:samples])
    
            if off_graph:
                # limit max number of samples tested
                test_values = dict(shuffled_values[samples:max_test_samples+1])
            else: 
                test_values = dict(shuffled_values[:samples])

            boards = value_dict.keys()
            num_boards = len(boards)

            logger.info("kept %i board samples", num_boards)

            index = dict(zip(boards, xrange(num_boards)))
            avectors_ND = numpy.array(map(specmine.go.board_to_affinity, boards))
            affinity_NN = specmine.discovery.affinity_graph(avectors_ND, neighbors, sigma = 1e6)

            for B in numpy.r_[0:300:10j].round().astype(int):
                if interpolate:
                    yield (run_template_features, [2, 2, B, test_values])
                    yield (run_random_features, [B, avectors_ND, index, test_values, interpolate], dict(aff_map = affinity_map))
                    #yield (run_laplacian_features, ["Laplacian",B,avectors_ND, affinity_NN, index, test_values, interpolate], dict(aff_map = affinity_map))
                    yield (run_clustered_graph_features, ["affinity", B, avectors_ND, affinity_NN, index, test_values, \
                        num_clusters,interpolate], dict(aff_map = affinity_map))
                else:
                    yield (run_template_features, [2, 2, B, test_values])
                    yield (run_random_features, [B, avectors_ND, index, test_values, interpolate])
                    #yield (run_laplacian_features, ["Laplacian",B,avectors_ND, affinity_NN, index, test_values, interpolate])
                    yield (run_graph_features, ["gameplay", B, avectors_ND, gameplay_NN, gameplay_index, test_values, num_clusters, interpolate])
                    #yield (run_clustered_graph_features, ["affinity", B, avectors_ND, affinity_NN, index, test_values, num_clusters, interpolate])

    with open(out_path, "wb") as out_file:
        writer = csv.writer(out_file)

        writer.writerow(["map_name", "features", "samples", "score_mean", "score_variance"])

        for (_, row) in condor.do(yield_jobs(), workers):
            writer.writerow(row)


@specmine.annotations(
    out_path = ("path to write CSV",),
    games_path = ("path to adjacency dict"),
    values_path = ("path to value function",),
    neighbors = ("number of neighbors", "option", None, int),
    workers = ("number of condor jobs", "option", None, int),
    ) 
def flat_affinity_test(out_path, games_path, values_path, neighbors = 5, workers = 0, interpolate = True, off_graph = True):
    """Test value prediction in Go."""

    value_list = get_value_list(games_path,values_path) 

    logger.info("number of value samples total: %i", len(value_list))
 
    def yield_jobs():
        min_samples = 5000
        max_samples = 10000
        step_samples = 5000
        max_test_samples = 100000

        shuffled_values = sorted(value_list, key = lambda _: numpy.random.rand()) 

        for samples in xrange(min_samples,max_samples,step_samples):
            # randomly sample subset of games 
            value_dict = dict(shuffled_values[:samples])
                
            # if testing off-graph use held-out samples
            if off_graph:
                test_values = dict(shuffled_values[samples:max_test_samples+1])
            else: 
                test_values = dict(shuffled_values[:samples])
            print type(test_values)

            boards = value_dict.keys()
            num_boards = len(boards)

            logger.info("kept %i board samples", num_boards)

            index = dict(zip(boards, xrange(num_boards)))
            avectors_ND = numpy.array(map(specmine.go.board_to_affinity, boards))
            affinity_NN = specmine.discovery.affinity_graph(avectors_ND, neighbors, sigma = 1e6)

            for B in numpy.r_[0:200:8j].round().astype(int):
                if interpolate:
                    yield (run_template_features, [2, 2, B, test_values])
                    yield (run_random_features, [B, avectors_ND, index, test_values, interpolate], dict(aff_map = affinity_map))
                    yield (run_laplacian_features, ["Laplacian",B,avectors_ND, affinity_NN, index, test_values, interpolate], dict(aff_map = affinity_map))
                else:
                    yield (run_template_features, [2, 2, B, test_values])
                    yield (run_random_features, [B, avectors_ND, index, test_values, interpolate])
                    yield (run_laplacian_features, ["Laplacian",B,avectors_ND, affinity_NN, index, test_values, interpolate])


    with open(out_path, "wb") as out_file:
        writer = csv.writer(out_file)

        writer.writerow(["map_name", "features", "samples", "score_mean", "score_variance"])

        for (_, row) in condor.do(yield_jobs(), workers):
            writer.writerow(row)

if __name__ == "__main__":
    specmine.script(flat_affinity_test)
    #specmine.script(clustered_affinity_test)
 
