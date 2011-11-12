import plac
import specmine

if __name__ == '__main__':
    plac.call(specmine.tools.write_ttt_states.main)

import json
import cPickle as pickle
import numpy
import tictac

def construct_adjacency(init_board = None, cutoff = None):
    states = {}

    if init_board is None:
        init_board = BoardState()

    def board_recurs(player, parent, board, depth = 0):
        print len(states)

        if cutoff is not None and depth == cutoff:
            return

        adjacent = states.get(board)

        if adjacent is None:
            adjacent = states[board] = set()

        if parent is not None:
            adjacent.add(parent)

        if board.check_end():
            return

        for i in xrange(3):
            for j in xrange(3):
                if board._grid[i,j] == 0:
                    board_recurs(-1 * player, board, board.make_move(player, i, j), depth + 1)

    board_recurs(1, None, init_board)

    return states

@plac.annotations(
    out_path = ("path to write states pickle",),
    start = ("start state", "option", None, json.loads),
    cutoff = ("move limit", "option", None, int),
    )
def main(out_path, start = None, cutoff = None):
    """Generate and write the TTT board adjacency map."""

    if start is None:
        start_board = tictac.BoardState()
    else:
        start_board = tictac.BoardState(numpy.array(start))

    states = tictac.construct_adjacency(start_board, cutoff = cutoff)

    with specmine.util.openz(out_path, "wb") as pickle_file:
        pickle.dump(states, pickle_file)

