import csv
import numpy
import cPickle as pickle
import condor
import specmine

logger = specmine.get_logger(__name__)

def affinity_map(state):
    ''' takes a Go State and returns the affinity map representation -- currently
    the flattened board'''
    return state.board.grid.flatten()


def run_features(map_name, B, all_features_NF, index, values):
    feature_map = specmine.discovery.TabularFeatureMap(all_features_NF, index)
    (mean, variance) = specmine.science.score_features_predict(feature_map, values)

    logger.info("with %i %s features, mean score is %.4f", B, map_name, mean)

    return [map_name, B, mean, variance]

def run_graph_features(map_name, B, vectors_ND, affinity_NN, index, values):
    if B > 0:
        basis_NB = specmine.spectral.laplacian_basis(affinity_NN, B)
        all_features_NF = numpy.hstack([vectors_ND, basis_NB])
    else:
        all_features_NF = vectors_ND

    return run_features(map_name, B, all_features_NF, index, values)

def run_random_features(B, vectors_ND, index, values):
    (N, _) = vectors_ND.shape

    random_basis_NB = numpy.random.random((N, B))
    all_features_NF = numpy.hstack([vectors_ND, random_basis_NB])

    return run_features("random", B, all_features_NF, index, values)

@specmine.annotations(
    out_path = ("path to write CSV",),
    games_path = ("path to adjacency dict"),
    values_path = ("path to value function",),
    neighbors = ("number of neighbors", "option", None, int),
    workers = ("number of condor jobs", "option", None, int),
    samples = ("(max) number of boards to evaluate from dataset", "option", None, int)
    )
def main(out_path, games_path, values_path, neighbors = 8, workers = 0, samples = 10000):
    """Test value prediction in Go."""
     
    games_path = specmine.util.static_path(games_path)
    values_path = specmine.util.static_path(values_path)

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

    value_list = sorted(value_list, key = lambda _: numpy.random.rand())
    value_list = value_list[:samples]
    value_dict = dict(value_list)
    
    boards = value_dict.keys()
    num_boards = len(boards)
    print 'number of samples kept: ', num_boards

    index = dict(zip(boards,xrange(num_boards)))
    avectors_ND = numpy.array(map(specmine.go.board_to_affinity, boards))
    affinity_NN = specmine.discovery.affinity_graph(avectors_ND, neighbors, sigmas = 2.0)
    print affinity_NN
    import sklearn.cluster
    spectral = sklearn.cluster.SpectralClustering(mode = "arpack")
    spectral.fit(affinity_NN)
    spectral = sklearn.cluster.SpectralClustering(mode = "amg")
    spectral.fit(affinity_NN)
    print "!!!"
    raise SystemExit()
    #basis_NB = specmine.spectral.laplacian_basis(affinity_NN, k = 32)
    #feature_map = specmine.discovery.InterpolationFeatureMap(basis_NB, avectors_ND, specmine.go.board_to_affinity)
#    feature_map = specmine.discovery.TabularFeatureMap(basis_NB, index)
#    score = specmine.science.score_features_predict(feature_map, value_dict)
    #print 'score: ', score

    def yield_jobs():
        for B in numpy.r_[0:20:5j].astype(int):
            #yield (run_random_features, [B, avectors_ND, index, values])
            #yield (run_graph_features, ["gameplay", B, avectors_ND, gameplay_NN, gameplay_index, value_dict])
            yield (run_graph_features, ["affinity", B, avectors_ND, affinity_NN, index, value_dict])

    with open(out_path, "wb") as out_file:
        writer = csv.writer(out_file)

        writer.writerow(["map_name", "features", "score_mean", "score_variance"])

        condor.do_or_distribute(yield_jobs(), workers, lambda _, r: writer.writerow(r))

if __name__ == "__main__":
    specmine.script(main)

