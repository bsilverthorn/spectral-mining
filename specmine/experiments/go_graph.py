import cPickle as pickle
import numpy
import specmine
import specmine.go

def main(num_games=10, num_samples=1e4):
    
    with specmine.util.openz(specmine.util.static_path\
      ('go_games/2010-01.pickle.gz')) as games_file:

        game_dict = pickle.load(games_file)
        print 'num_games: ', len(game_dict.values())
    # build the affinity representation

    all_grids = [game.grids for game in game_dict.itervalues()]
    affinity_vectors = numpy.vstack(all_grids)

    num_boards = affinity_vectors.shape[0]
    print 'number of boards: ', num_boards

    print 'affinity vector shape: ', affinity_vectors.shape
    print 'total num boards: ', num_boards
    print 'subsampling states'
    affinity_vectors = affinity_vectors[numpy.random.permutation(num_boards),:,:]
    affinity_vectors = affinity_vectors[:num_samples,:,:]
    affinity_vectors = numpy.reshape(affinity_vectors,(num_samples,81))

    print 'building affinity graph'
    graph_mat = specmine.discovery.affinity_graph(affinity_vectors,neighbors=5)

    # save this graph if it gets big?
    print "..."
    graph_dict = specmine.discovery.adjacency_matrix_to_dict(graph_mat)

    #specmine.graphviz.visualize_graph("go_graph_test.pdf", graph_dict, "twopi")
    specmine.graphviz.visualize_graph("go_graph_test.neato.pdf", graph_dict, "neato")

    # what to do with this graph?

if __name__ == "__main__":
    specmine.script(main)

