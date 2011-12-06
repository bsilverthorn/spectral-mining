import csv
import numpy
import cPickle as pickle
import condor
import specmine

logger = specmine.get_logger(__name__)

@specmine.annotations(
    out_path = ("path to write CSV",),
    games_path = ("path to adjacency dict"),
    #values_path = ("path to value function",),
    neighbors = ("number of neighbors", "option", None, int),
    workers = ("number of condor jobs", "option", None, int),
    )
def main(out_path, games_path, neighbors = 8, workers = 0):
    """Test value prediction in Go."""

    with specmine.util.openz(games_path) as games_file:
        games = pickle.load(games_file).values()

    boards = specmine.go.boards_from_games(games)
    avectors_ND = numpy.array(map(specmine.go.board_to_affinity, boards))
    affinity_NN = specmine.discovery.affinity_graph(avectors_ND, neighbors = neighbors)
    basis_NB = specmine.spectral.laplacian_basis(affinity_NN, k = 32)
    feature_map = specmine.discovery.InterpolationFeatureMap(basis_NB, avectors_ND, specmine.go.board_to_affinity)

    for x in xrange(9):
        for y in xrange(9):
            grid = numpy.zeros((9, 9), numpy.int8)

            grid[x, y] = 1

            board = specmine.go.BoardState(grid)

            print board, feature_map[board]

if __name__ == "__main__":
    specmine.script(main)

