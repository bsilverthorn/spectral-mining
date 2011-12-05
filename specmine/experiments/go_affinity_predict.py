import csv
import numpy
import cPickle as pickle
import condor
import specmine

logger = specmine.get_logger(__name__)

def raw_state_features((board, player)):
    return [1] + list(board.grid.flatten())

# XXX make into an evaluate-features class, load the opponent once, etc.

def run_features(map_name, B, all_features_NF, index, values):
    feature_map = specmine.discovery.TabularFeatureMap(all_features_NF, index)
    #(mean, variance) = specmine.science.score_features_predict(feature_map, values)
    (mean, variance) = specmine.science.score_features_regress_act(feature_map, values)

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
    states_path = ("path to adjacency dict", "option",),
    values_path = ("path to value function",),
    neighbors = ("number of neighbors", "option", None, int),
    workers = ("number of condor jobs", "option", None, int),
    )
def main(out_path, values_path, neighbors = 8, workers = 0):
    """Run TTT state-clustering experiment(s)."""

    # load the value function
    with specmine.openz(values_path) as values_file:
        values = pickle.load(values_file)

    # convert states to their vector representations
    states_adict = specmine.tictac.load_adjacency_dict()
    states = list(states_adict)

    logger.info("converting states to their vector representation")

    affinity_index = dict(zip(states, xrange(len(states))))
    vectors_ND = numpy.array(map(raw_state_features, states))

    # build the affinity graph
    affinity_NN = specmine.discovery.affinity_graph(vectors_ND, neighbors)
    (gameplay_NN, gameplay_index) = specmine.discovery.adjacency_dict_to_matrix(states_adict)

    def yield_jobs():
        for B in numpy.r_[0:400:64j].astype(int):
            yield (run_random_features, [B, vectors_ND, affinity_index, values])
            yield (run_graph_features, ["gameplay", B, vectors_ND, gameplay_NN, gameplay_index, values])
            yield (run_graph_features, ["affinity", B, vectors_ND, affinity_NN, affinity_index, values])

    with open(out_path, "wb") as out_file:
        writer = csv.writer(out_file)

        writer.writerow(["map_name", "features", "score_mean", "score_variance"])

        condor.do_or_distribute(yield_jobs(), workers, lambda _, r: writer.writerow(r))

if __name__ == "__main__":
    specmine.script(main)

