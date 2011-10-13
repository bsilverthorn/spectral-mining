import json
import plac
import numpy
import tictac

def ttt_value_min(board, player, alpha = -numpy.inf, beta = numpy.inf):
    """Compute the state value (min node) with alpha-beta pruning."""

    # terminal state?
    winner = board.get_winner()

    if winner is not None:
        return -1 * player * winner

    # no; recurse
    min_value = numpy.inf

    for i in xrange(3):
        for j in xrange(3):
            if board._grid[i, j] == 0:
                value = ttt_value_max(board.make_move(player, i, j), -1 * player, alpha, beta)

                if value <= alpha:
                    return value

                if min_value > value:
                    min_value = value

                    if beta > value:
                        beta = value

    return min_value

def ttt_value_max(board, player, alpha = -numpy.inf, beta = numpy.inf):
    """Compute the state value (max node) with alpha-beta pruning."""

    # terminal state?
    winner = board.get_winner()

    if winner is not None:
        return player * winner

    # no; recurse
    max_value = -numpy.inf

    for i in xrange(3):
        for j in xrange(3):
            if board._grid[i, j] == 0:
                value = ttt_value_min(board.make_move(player, i, j), -1 * player, alpha, beta)

                if value >= beta:
                    return value

                if max_value < value:
                    max_value = value

                    if alpha < value:
                        alpha = value

    return max_value

def ttt_optimal_move(board, player):
    """Compute the optimal move (max node)."""

    max_i = None
    max_j = None
    max_value = -numpy.inf

    for i in xrange(3):
        for j in xrange(3):
            if board._grid[i, j] == 0:
                value = ttt_value_min(board.make_move(player, i, j), -1 * player)

                if max_value < value:
                    max_i = i
                    max_j = j
                    max_value = value

    return (max_i, max_j, max_value)

def test_ttt_minimax():
    configurations = [
        [[-1,  0,  0],
         [ 0, -1,  1],
         [ 1,  0,  0]],
        [[ 1, -1, -1],
         [-1,  1,  1],
         [ 1,  0, -1]],
        [[-1,  1, -1],
         [-1,  1,  1],
         [ 0,  0, -1]],
        [[ 0,  0,  1],
         [-1, -1, -1],
         [ 0,  1,  0]],
        ]
    boards = [tictac.BoardState(numpy.array(c)) for c in configurations]

    assert ttt_optimal_move(boards[0], 1) == (2, 2, 1)
    assert ttt_optimal_move(boards[1], 1) == (2, 1, 0)
    assert ttt_optimal_move(boards[2], 1) == (2, 1, 1)
    assert ttt_optimal_move(boards[2], -1) == (2, 0, 1)
    assert ttt_value_max(boards[3], -1) == 1

@plac.annotations(
    board = ("board state (JSON)", "positional", None, json.loads),
    player = ("player (-1 or 1)", "option", "p", int),
    )
def main(board, player = 1):
    """Print the optimal Tic-Tac-Toe move in a given state."""

    print ttt_optimal_move(tictac.BoardState(numpy.array(board)), player)

if __name__ == "__main__":
    plac.call(main)

