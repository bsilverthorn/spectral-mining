import specmine.tools.write_ttt_affinity

if __name__ == '__main__':
    specmine.script(specmine.tools.write_ttt_affinity.main)

import cPickle as pickle
import numpy
import sklearn.neighbors
import specmine

logger = specmine.get_logger(__name__)

def raw_state_features((board, player)):
    return [1] + list(board.grid.flatten())

@specmine.annotations(
    out_path = ("path to write states pickle",),
    )
def main(out_path, neighbors = 4):
    """Generate and write the TTT affinity-graph adjacency dict."""

    states = list(specmine.tictac.load_adjacency_dict())

    logger.info("converting states to their vector representation")

    vectors_ND = numpy.array(map(raw_state_features, states))
    tree = sklearn.neighbors.BallTree(vectors_ND)
    (neighbor_distances_NG, neighbor_indices_NG) = tree.query(vectors_ND, k = neighbors)
    adjacency_dict = {}

    for n in xrange(len(states)):
        state = states[n]
        neighbors = adjacency_dict.get(state)

        if neighbors is None:
            adjacency_dict[state] = neighbors = []

        neighbors.extend(states[m] for m in neighbor_indices_NG[n, :] if m != n)

    with specmine.util.openz(out_path, "wb") as out_file:
        pickle.dump(adjacency_dict, out_file)

