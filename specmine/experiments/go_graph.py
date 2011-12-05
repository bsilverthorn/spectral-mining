import cPickle as pickle
import numpy
import specmine
import specmine.go

def main(num_games=10, num_samples=1e5):
    
    with specmine.util.openz(specmine.util.static_path\
      ('go_games/2010-01.pickle.gz')) as games_file:

        game_dict = pickle.load(games_file)
        print 'num_games: ', len(game_dict.values())
    # build the affinity representation

    all_grids = [game.grids for game in game_dict.itervalues()]
    affinity_vectors = numpy.vstack(all_grids)

    num_boards = affinity_vectors.shape[0]
    print 'number of boards: ', num_boards

    affinity_vectors = numpy.reshape(affinity_vectors,(num_boards,81))
    print 'affinity vector shape: ', affinity_vectors.shape
    print 'total num boards: ', num_boards
    print 'subsampling states'
    affinity_vectors = affinity_vectors[numpy.random.permutation(num_boards),:]
    affinity_vectors = affinity_vectors[:num_samples]

    print 'building affinity graph'
    graph = specmine.discovery.affinity_graph(affinity_vectors,neighbors=5)
    print graph[0]
    print graph.shape
    # what to do with this graph?

if __name__ == "__main__":
    main()
