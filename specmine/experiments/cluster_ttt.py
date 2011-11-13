import plac
import specmine

if __name__ == "__main__":
    plac.call(specmine.experiments.cluster_ttt)

import cPickle as pickle
import numpy
import scipy.sparse

def raw_state_features((board, player)):
    return board.flatten()

@plac.annotations()
def main():
    states_path = specmine.util.static_path("ttt_states.pickle.gz")

    with specmine.util.openz(states_path) as pickle_file:
        boards = pickle.load(pickle_file)

    # construct the adjacency matrix
    K = 8
    N = len(boards)
    index = dict(zip(boards, xrange(N)))
    rindex = sorted(index, key = index.__getitem__)
    adjacency = scipy.sparse.lil_matrix((N, N))

    for board in boards:
        n = index[board]
        affinities = numpy.fromiter(hamming_affinity(board, b) for b in boards, float)
        ordered = numpy.argsort(affinities)

        for i in ordered[-K:]:
            adjacency[n, index[]] = affinities[m]

