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

def run_features(map_name, bases, feature_map, values):
    logger.info("scoring %i %s feature(s) on %i examples", bases, map_name, len(values))

    (mean, variance) = specmine.science.score_features_predict(feature_map, values)

    logger.info("with %i %s feature(s), mean score is %.4f", bases, map_name, mean)

    return [map_name, bases, mean, variance]

def run_graph_features(map_name, bases, avectors_ND, values):
    if bases > 0:
        (k_means, blocks) = specmine.spectral.clustered_basis_from_affinity(avectors_ND, bases)

        maps = []

        for (k, (basis, tree)) in enumerate(blocks):
            full_basis = numpy.hstack([avectors_ND[k_means.labels_ == k], basis])
            map_ = specmine.discovery.InterpolationFeatureMapRaw(full_basis, tree, affinity_map)

            maps.append(map_)

        global_map = specmine.discovery.ClusteredFeatureMap(affinity_map, k_means, maps)
    else:
        # XXX still requires full-set balltree construction
        global_map = specmine.discovery.InterpolationFeatureMap(avectors_ND, avectors_ND, affinity_map)

    return run_features(map_name, bases, global_map, values)

def run_random_features(bases, avectors_ND, values):
    B = bases
    (N, _) = avectors_ND.shape

    random_basis_NB = numpy.random.random((N, B))
    all_features_NF = numpy.hstack([avectors_ND, random_basis_NB])

    return run_features("random", bases, all_features_NF, avectors_ND, values)

def get_value_list(games_path,values_path):
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
def main(out_path, games_path, values_path, neighbors = 8, workers = 0, off_graph = True):
    """Test value prediction in Go."""

    value_list = get_value_list(games_path, values_path)

    logger.info("number of value samples total: %i", len(value_list))
 
    def yield_jobs():
        samples = 50000

        shuffled_values = sorted(value_list, key = lambda _: numpy.random.rand()) 

        # randomly sample subset
        value_dict = dict(shuffled_values[:samples])

        test_samples = 20000
        if off_graph:
            test_values = dict(shuffled_values[-test_samples:])
        else: 
            test_values = dict(shuffled_values[:test_samples])

        logger.info("kept %i board samples", len(value_dict))

        avectors_ND = numpy.array(map(specmine.go.board_to_affinity, value_dict))

        for B in numpy.r_[1:200:8j].round().astype(int):
            yield (run_graph_features, ["affinity", B, avectors_ND, test_values])

    with open(out_path, "wb") as out_file:
        writer = csv.writer(out_file)

        writer.writerow(["map_name", "features", "score_mean", "score_variance"])

        for (_, row) in condor.do(yield_jobs(), workers):
            writer.writerow(row)

            out_file.flush()

if __name__ == "__main__":
    specmine.script(main)
 
