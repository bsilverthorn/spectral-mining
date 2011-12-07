import numpy
import specmine

logger = specmine.get_logger(__name__)

def grid_to_affinity(grid):
    return grid.flatten()

def board_to_affinity(board):
    return grid_to_affinity(board.grid)

def boards_from_games(games, samples = 10000):
    """Extract boards from games and return a subset."""

    grids = numpy.vstack([game.grids for game in games])
    boards = set(map(specmine.go.BoardState, grids))
    shuffled = sorted(boards, key = lambda _: numpy.random.rand())

    return shuffled[:samples]

