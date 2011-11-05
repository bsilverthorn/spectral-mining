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

    return scipy.sparse.coo_matrix((coo_values, (coo_is + coo_js, coo_js + coo_is)))

@plac.annotations(
    out_path = ("path to write clustering",),
    states_path = ("path to states pickle",),
    )
def main(out_path, states_path):
    """Cluster a state space graph."""

    with specmine.util.openz("ttt_states.pickle.gz") as pickle_file:
        boards = pickle.load(pickle_file)

    adjacency = state_dict_to_adjacency(boards)
    clustering_array = specmine.graclus.cluster(adjacency, 4)
    clustering_dict = dict(zip(boards, clustering_array))

    with open(out_path, "wb") as out_file:
        pickle.dump(clustering_dict, out_file)

