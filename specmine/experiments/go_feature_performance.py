import csv
import numpy
import cPickle as pickle
import condor
import specmine

logger = specmine.get_logger(__name__)

@specmine.annotations(
    out_path = ("path to write CSV",),
    games_path = ("path to adjacency dict"),
    values_path = ("path to value function",),
    workers = ("number of condor jobs", "option", None, int),
    neighbors = ("number of neighbors", "option", None, int),
    interpolate = ("interpolate to off-graph points?", "option", None, bool),
    min_samples = ("min number of samples used for graph features", "option", None, int),
    max_samples = ("max number of samples used for graph features", "option", None, int),
    step_samples = ("ammount to increment sample size by", "option", None, int),
    max_test_samples = ("max number of samples used for testing value prediction error", "option", None, int),
    max_num_features = ("max number of features used", "option", None, int)
    )
def measure_feature_performance( \
    out_path, games_path, values_path,  workers = 0,\
    neighbors = 8, interpolate = True, \
    min_samples = 5000, max_samples = 5000, step_samples = 5000, max_test_samples = 250000, \
    max_num_features=100):
    
    values = get_value_list(games_path,values_path)
    values = sorted(values, key = lambda _: numpy.random.rand()) # shuffle values

    def yield_jobs():
        for samples in xrange(min_samples, max_samples+step_samples, step_samples):

            value_dict = dict(values[:samples])
            # if testing off-graph use held-out samples
            if interpolate:
                test_values = dict(values[samples:max_test_samples+1])
            else: 
                test_values = dict(values[:samples])

            boards = value_dict.keys()
            num_boards = len(boards)

            logger.info("kept %i board samples", num_boards)

            index = dict(zip(boards, xrange(num_boards)))
            avectors_ND = numpy.array(map(specmine.feature_maps.affinity_map, boards))
            # TODO check affinity graph construction
            affinity_NN = specmine.discovery.affinity_graph(avectors_ND, neighbors, sigma = 1e6)

            for NF in numpy.r_[0:max_num_features:10j].round().astype(int):
                if interpolate:
                    yield (run_template_features, [2, 2, NF, test_values])
                    yield (run_laplacian_features, ["Laplacian", NF, avectors_ND, affinity_NN, index, test_values, interpolate], dict(aff_map = specmine.feature_maps.affinity_map))
                    # yield (run_random_features, [B, avectors_ND, index, test_values, interpolate], dict(aff_map = specmine.feature_maps.affinity_map))

                else:
                    yield (run_template_features, [2, 2, NF, test_values])
                    yield (run_laplacian_features, ["Laplacian", NF, avectors_ND, affinity_NN, index, test_values, interpolate])
                    # yield (run_random_features, [B, avectors_ND, index, test_values, interpolate])

    with open(out_path, "wb") as out_file:
        writer = csv.writer(out_file)

        writer.writerow(["map_name", "features", "samples", "score_mean", "score_variance"])

        for (_, row) in condor.do(yield_jobs(), workers):
            writer.writerow(row)

def run_template_features(m,n,B,values):

    feature_map = specmine.feature_maps.TemplateFeatureMap(m,n,B)
    (mean, variance) = specmine.science.score_features_predict(feature_map, values)
    
    m,n = feature_map.grids[0].shape
    return [str(m)+'by'+str(n)+"template", feature_map.B, None ,mean, variance]

def run_proximity_features(map_name, B, all_features_NF, affinity_vectors, index, values, interpolate = False, **kwargs):
    
    if interpolate:    
        aff_map = kwargs.get('aff_map')
        assert aff_map is not None

        feature_map = specmine.feature_maps.InterpolationFeatureMap(all_features_NF, affinity_vectors, aff_map)

    else:
        feature_map = specmine.feature_maps.TabularFeatureMap(all_features_NF, index)

    (mean, variance) = specmine.science.score_features_predict(feature_map, values)

    logger.info("with %i %s features, mean score is %.4f", B, map_name, mean)

    num_samples = len(index)#feature_map.basis.shape[0]
    if B >0:
        assert len(index) == feature_map.basis.shape[0]

    return [map_name, B, num_samples, mean, variance]

def run_laplacian_features(map_name, B, vectors_ND, affinity_NN, index, values, interpolate = False, **kwargs):
    if B > 0:
        basis_NB = specmine.spectral.laplacian_basis(affinity_NN, B, method = "arpack")
    else:
        basis_NB = None
        
    return run_proximity_features(map_name, B, basis_NB, vectors_ND, index, values, interpolate, **kwargs)

def run_random_features(B, vectors_ND, index, values, interpolate = False, **kwargs):
    if B > 0:
        N = vectors_ND.shape[0]
        all_features_NF = numpy.random.random((N, B))
    else:
        all_features_NF = None

    return run_proximity_features("random", B, all_features_NF, vectors_ND, index, values, interpolate, **kwargs)


def get_value_list(games_path,values_path):
    ''' takes a list of games and a dictionary of values and builds a list of
    (BoardState, value) pairs '''

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

if __name__ == "__main__":
    specmine.script(measure_feature_performance)
