import numpy
import specmine

def graph_from_go_games(games_path, neighbors = 8, samples = 10000):

    # build the affinity representation
    all_grids = [game.grids for game in game_dict.itervalues()]
    affinity_vectors = numpy.vstack(all_grids)

    num_boards = affinity_vectors.shape[0]
    print 'number of boards: ', num_boards

    print 'affinity vector shape: ', affinity_vectors.shape
    print 'total num boards: ', num_boards
    print 'subsampling states'
    # remove duplicates
    board_states = set(map(specmine.go.BoardState, affinity_vectors))
    affinity_vectors = numpy.array([b.grid for b in board_states])
    #subsample
    numpy.random.shuffle(affinity_vectors)

    affinity_vectors = affinity_vectors[:samples, ...]
    # reshape
    affinity_vectors = numpy.reshape(affinity_vectors,(len(affinity_vectors),81))

    B = 32
    affinity_NN = specmine.discovery.affinity_graph(affinity_vectors, neighbors=8)
    basis_NB = specmine.spectral.laplacian_basis(affinity_NN, B)

    return basis_NB

