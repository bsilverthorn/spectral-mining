import numpy
import sklearn.neighbors
import specmine
import go

def main(num_games=10, num_samples=1e5):
    
    game_dict = specmine.util.openz(specmine.util.satic_path('go_games/2010-01.pickle.gz'))
    
    # build the affinity representation

    affinity_vectors = numpy.zeros((0,9,9))
    for game in game_dict.values():
        numpy.vstack(affinity_vectors,game.grids)

    num_boards = affinity_vectors.shape[0]
    print 'number of boards: ', num_boards

    affinity_vectors = numpy.reshape(affinity_vectors,(num_boards,81))
    affinity_vectors = affinity_vectors[numpy.random.permutation(num_boards),:]
    affinity_vectors = affinity_vectors[:num_samples]

    graph = specmine.discovery.affinity_graph(affinity_vectors)
    # what to do with this graph?

