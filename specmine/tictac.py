import os.path
import gzip
import csv
import cPickle as pickle
import contextlib
import numpy
import scipy.sparse
import scipy.sparse.linalg
import spectral

class BoardState(object):
    # XXX use a simple bitstring board representation?
    # XXX an efficient end-state check could be made against a mask set

    def __init__(self, grid = None):
        if grid is None:
            grid = numpy.zeros((3, 3))

        self._string = str(grid)
        self._grid = grid

    def __hash__(self):
        return hash(self._string)

    def __eq__(self, other):
        return self._string == other._string

    def __str__(self):
        return self._string

    def __repr__(self):
        return repr(self._grid)

    def make_move(self, player, i, j):
        assert self._grid[i, j] == 0

        new_grid = self._grid.copy()

        new_grid[i,j] = player

        return BoardState(new_grid)

    def get_winner(self):
        """Return the winner of the board, if any."""

        # implemented explicitly for easy Cythonization

        # draw?
        board_product = 1

        for i in xrange(3):
            board_product *= self._grid[i, 0] * self._grid[i, 1] * self._grid[i, 2]

        if board_product != 0:
            return 0

        # row or column win?
        for i in xrange(3):
            row_sum = self._grid[i, 0] + self._grid[i, 1] + self._grid[i, 2]
            col_sum = self._grid[0, i] + self._grid[1, i] + self._grid[2, i]

            if row_sum == 3 or col_sum == 3:
                return 1
            elif row_sum == -3 or col_sum == -3:
                return -1

        # diagonal win?
        tb_diag_sum = self._grid[0, 0] + self._grid[1, 1] + self._grid[2, 2]
        bt_diag_sum = self._grid[0, 2] + self._grid[1, 1] + self._grid[2, 0]

        if tb_diag_sum == 3 or bt_diag_sum == 3:
            return 1
        elif tb_diag_sum == -3 or bt_diag_sum == -3:
            return -1

        # no win, game still in progress
        return None

    def get_player(self):
        """Return the player to act on the board."""

        if numpy.sum(numpy.abs(self._grid)) % 2 == 0:
            return 1
        else:
            return -1

    def check_end(self):
        """Is this board state an end state?"""

        return self.get_winner() != None

    @property
    def grid(self):
        return self._grid

def construct_adjacency_dict(init_board = None, cutoff = None):
    states = {}

    if init_board is None:
        init_board = BoardState()

    def board_recurs(player, parent, board, depth = 0):
        print len(states)

        if cutoff is not None and depth == cutoff:
            return

        adjacent = states.get(board)

      for i in xrange(3):
            for j in xrange(3):
                if board._grid[i,j] == 0:
                    board_recurs(-1 * player, board, board.make_move(player, i, j), depth + 1)

    board_recurs(1, None, init_board)

    return states
