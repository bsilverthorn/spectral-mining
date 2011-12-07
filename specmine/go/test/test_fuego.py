import numpy
import nose.tools
import specmine

def test_fuego_board_basic():
    board = specmine.go.FuegoBoard()

    nose.tools.assert_equal(board.size, 9)
    nose.tools.assert_equal(board.to_play, 1)
    nose.tools.assert_true(numpy.all(board.grid == 0))

    board.play(0, 8)

    nose.tools.assert_equal(board.to_play, -1)
    nose.tools.assert_equal(board[0, 8], 1)
    nose.tools.assert_equal(numpy.sum(numpy.abs(board.grid)), 1)

    board.play(8, 1)

    nose.tools.assert_equal(board.to_play, 1)
    nose.tools.assert_equal(board[0, 8], 1)
    nose.tools.assert_equal(board[8, 1], -1)
    nose.tools.assert_equal(numpy.sum(numpy.abs(board.grid)), 2)
    nose.tools.assert_false(board.is_legal(0, 8))
    nose.tools.assert_false(board.is_legal(8, 1))

    board.initialize()

    nose.tools.assert_equal(board.size, 9)
    nose.tools.assert_equal(board.to_play, 1)
    nose.tools.assert_true(numpy.all(board.grid == 0))

def test_fuego_moves_to_grids():
    moves = [
        (1, 4, 4),
        (-1, 3, 3),
        (1, 2, 2),
        ]
    grids = specmine.go.moves_to_grids(moves)

    nose.tools.assert_equal(grids[0, 2, 2], 0)
    nose.tools.assert_equal(grids[0, 3, 3], 0)
    nose.tools.assert_equal(grids[0, 4, 4], 1)
    nose.tools.assert_equal(grids[1, 3, 3], -1)
    nose.tools.assert_equal(grids[2, 2, 2], 1)

def test_fuego_moves_to_board():
    moves = [
        (1, 4, 4),
        (-1, 3, 3),
        (1, 2, 2),
        ]
    board = specmine.go.moves_to_board(moves)

    nose.tools.assert_equal(board[2, 2], 1)
    nose.tools.assert_equal(board[3, 3], -1)
    nose.tools.assert_equal(board[4, 4], 1)
    nose.tools.assert_equal(numpy.sum(numpy.abs(board.grid)), 3)

def test_fuego_random_player_basic():
    board = specmine.go.FuegoBoard()
    player = specmine.go.FuegoRandomPlayer(board)

    moves_a = [player.generate_move() for _ in xrange(16)]
    moves_b = [player.generate_move() for _ in xrange(16)]

    nose.tools.assert_not_equal(moves_a, moves_b)

def test_fuego_board_score_simple_endgame():
    board = specmine.go.FuegoBoard()
    player = specmine.go.FuegoRandomPlayer(board)
    next_player = specmine.go.FuegoCapturePlayer(board)

    while True:
        move = player.generate_move()

        (player, next_player) = (next_player, player)

        if move is None:
            break
        else:
            (row, column) = move

            board.play(row, column)

    print board.score_simple_endgame()

