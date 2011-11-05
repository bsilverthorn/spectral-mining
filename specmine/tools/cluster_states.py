import plac
import specmine

if __name__ == "__main__":
    plac.call(specmine.tools.cluster_states.main)

import cPickle as pickle
import numpy
import scipy.sparse

def state_dict_to_adjacency(states):
    """Build a (symmetric) adjacency matrix from a {state: [state]} dict."""

    indices = dict(zip(states, xrange(len(states))))
    coo_is = []
    coo_js = []

    for (board, next_boards) in states.iteritems():
        for next_board in next_boards:
            coo_is.append(indices[board])
            coo_js.append(indices[next_board])

    coo_values = numpy.ones(len(coo_is) * 2, int)

    return (
        indices,
        scipy.sparse.coo_matrix((coo_values, (coo_is + coo_js, coo_js + coo_is))),
        )

@plac.annotations(
    out_path = ("path to write clustering",),
    states_path = ("path to states pickle",),
    clusters = ("number of clusters", "positional", None, int),
    )
def main(out_path, states_path, clusters):
    """Cluster a state space graph."""

    with specmine.util.openz(states_path) as pickle_file:
        boards = pickle.load(pickle_file)

    (indices, adjacency) = state_dict_to_adjacency(boards)
    clustering_array = specmine.graclus.cluster(adjacency, clusters)
    clustering_dict = dict((s, clustering_array[indices[s]]) for s in boards)

    with open(out_path, "wb") as out_file:
        pickle.dump(clustering_dict, out_file)
