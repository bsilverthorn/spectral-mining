import plac
import specmine

if __name__ == "__main__":
    plac.call(specmine.experiments.cluster_ttt)

import cPickle as pickle
import numpy
import scipy.sparse

def hamming_affinity(board_a, board_b):
    return numpy.sum(board_a == board_b)

@plac.annotations()
def main():
    states_path = specmine.util.static_path("ttt_states.pickle.gz")

    with specmine.util.openz(states_path) as pickle_file:
        boards = pickle.load(pickle_file)

    # construct the adjacency matrix
    N = len(boards)
    index = dict(zip(boards, xrange(N)))
    adjacency = scipy.sparse.lil_matrix((N, N))

    for board in boards:
        n = index[board]
        distances = [hamming_affinity(board, b) for b in boards]

