import numpy
import specmine

logger = specmine.get_logger(__name__)

def grid_to_affinity(grid):
    return grid.flat

def board_to_affinity(board):
    return grid_to_affinity(board.grid)

def boards_from_games(games, samples = 10000):
    """Extract boards from games and return a subset."""

    grids = numpy.vstack([game.grids for game in games])
    boards = set(map(specmine.go.BoardState, grids))
    shuffled = sorted(boards, key = lambda _: numpy.random.rand())

    return shuffled[:samples]

def graph_from_games(games, neighbors = 8, samples = 10000):
    logger.info("removing duplicate boards")

    affinity_vectors = numpy.array(map(board_to_affinity, sampled_boards))

    return specmine.discovery.affinity_graph(affinity_vectors, neighbors = neighbors)

