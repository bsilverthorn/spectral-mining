import cPickle as pickle
import numpy
import specmine

@specmine.annotations(
    out_path = ("path to write visualization",),
    games_path = ("path to adjacency dict"),
    render_with = ("graphviz rendering tool", "option", "r"),
    eigenvector = ("color according to an eigenvector", "option", "g", int),
    samples = ("number of boards to sample", "option", None, int),
    neighbors = ("number of neighbors in graph", "option", None, int),
    largest = ("largest eigenvectors?", "flag"),
    )
def main(
    out_path,
    games_path,
    render_with = "neato",
    eigenvector = None,
    samples = 10000,
    neighbors = 8,
    largest = False,
    ):
    """Visualize a state space graph in Go."""

    with specmine.util.openz(games_path) as games_file:
        games = pickle.load(games_file).values()

    boards = specmine.go.boards_from_games(games, samples = samples)
    boards = sorted(boards, key = lambda _: numpy.random.rand())
    avectors_ND = numpy.array(map(specmine.go.board_to_affinity, boards))
    affinity_NN = specmine.discovery.affinity_graph(avectors_ND, neighbors = neighbors, sigma = 1e16)
    graph_dict = specmine.discovery.adjacency_matrix_to_dict(affinity_NN, make_directed = True)

    if eigenvector is None:
        coloring = None
    else:
        basis_NB = specmine.spectral.laplacian_basis(affinity_NN, largest = largest, k = eigenvector + 1)
        colors = specmine.graphviz.continuous_to_coloring(basis_NB[:, eigenvector])
        coloring = dict(zip(numpy.arange(samples), colors))

    specmine.graphviz.visualize_graph(out_path, graph_dict, render_with, coloring = coloring)

if __name__ == "__main__":
    specmine.script(main)

