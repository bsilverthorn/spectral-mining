import numpy
import nose.tools
import specmine

def test_board_state():
    configurations = [
        [[0, 0, 0], [0, 0, 0], [0, 0, 0]],
        [[1, 0, 0], [0, 1, 0], [0, 0, 1]],
        [[0, -1, 0], [0, -1, 0], [0, -1, 0]],
        [[0, 0, 1], [-1, -1, -1], [0, 1, 0]],
        [[1, 1, -1], [-1, 1, -1], [1, -1, 1]],
        ]
    boards = [specmine.tictac.BoardState(numpy.array(c)) for c in configurations]

    assert boards[0].get_winner() is None
    assert boards[0].make_move(1, 2, 2)._grid[2, 2] == 1
    assert boards[0]._grid[2, 2] == 0
    assert boards[0] == boards[0]
    assert boards[0] != boards[1]
    assert boards[1].get_winner() == 1
    assert boards[2].get_winner() == -1
    assert boards[3].get_winner() == -1
    assert boards[4].get_winner() == 0

def test_construct_adjacency_dict():
    start = specmine.tictac.BoardState()
    adict = specmine.tictac.construct_adjacency_dict(cutoff = 2)
    all_first = set((start.make_move(1, i, j), -1) for i in xrange(3) for j in xrange(3))

    nose.tools.assert_equal(adict[(start, 1)], all_first)

    for first in all_first:
        nose.tools.assert_equal(len(adict[first]), 8)

