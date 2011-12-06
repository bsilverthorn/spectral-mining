import numpy
import nose.tools
import specmine

def test_fuego_board_basic():
    board = specmine.go.FuegoBoard()

    nose.tools.assert_equal(board.to_play, 1)
    nose.tools.assert_true(numpy.all(board.grid == 0))

    board.play(0, 8)

    nose.tools.assert_equal(board.to_play, -1)
    nose.tools.assert_equal(board.grid[0, 8], 1)
    nose.tools.assert_equal(numpy.sum(numpy.abs(board.grid)), 1)

    board.play(8, 1)

    nose.tools.assert_equal(board.to_play, 1)
    nose.tools.assert_equal(board.grid[0, 8], 1)
    nose.tools.assert_equal(board.grid[8, 1], -1)
    nose.tools.assert_equal(numpy.sum(numpy.abs(board.grid)), 2)
    nose.tools.assert_false(board.is_legal(0, 8))
    nose.tools.assert_false(board.is_legal(8, 1))

