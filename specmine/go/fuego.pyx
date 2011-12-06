import numpy

cimport numpy
cimport specmine.go.fuego_c as fuego_c

cdef fuego_c.SgBlackWhite _player_to_black_white(int player):
    if player == 1:
        return fuego_c.SG_BLACK
    elif player == -1:
        return fuego_c.SG_WHITE
    else:
        assert False

cdef int _black_white_to_int(fuego_c.SgBlackWhite black_white):
    if black_white == fuego_c.SG_BLACK:
        return 1
    elif black_white == fuego_c.SG_WHITE:
        return -1
    else:
        assert False

cdef int _board_color_to_int(fuego_c.SgBoardColor color):
    if color == fuego_c.SG_EMPTY:
        return 0
    elif color == fuego_c.SG_BLACK:
        return 1
    elif color == fuego_c.SG_WHITE:
        return -1
    elif color == fuego_c.SG_BORDER:
        return 9
    else:
        assert False

cdef class FuegoBoard(object):
    cdef int _size
    cdef fuego_c.GoBoard* _board

    def __cinit__(self, int size = 9):
        self._size = size
        self._board = new fuego_c.GoBoard(size)

    def __init__(self, int size = 9):
        pass

    def __dealloc__(self):
        del self._board

    def is_legal(self, int row, int column):
        """Is the specified move legal?"""

        return self._board.IsLegal(self._row_column_to_point(row, column))

    def play(self, int row, int column):
        """Play a single move."""

        self._board.Play(self._row_column_to_point(row, column))

        self._assert_move_was_ok()

    @property
    def to_play(self):
        """Return the player to play."""

        return _black_white_to_int(self._board.ToPlay())

    @property
    def grid(self):
        """Array representation of the board state."""

        cdef numpy.ndarray[numpy.int8_t, ndim = 2] grid = numpy.empty((self._size, self._size), numpy.int8)

        for r in xrange(self._size):
            for c in xrange(self._size):
                grid[r, c] = self._at(r, c)

        return grid

    cdef int _at(self, int row, int column):
        """Return the integer board state at a position."""

        point = self._row_column_to_point(row, column)
        color = self._board.GetColor(point)

        return _board_color_to_int(color)

    cdef fuego_c.SgPoint _row_column_to_point(self, int row, int column):
        """Convert a row/column coordinate."""

        assert 0 <= row < self._size
        assert 0 <= column < self._size

        return fuego_c.Pt(column + 1, self._size - row)

    cdef int _point_to_row(self, fuego_c.SgPoint point):
        """Extract the row from a point."""

        return self._size - fuego_c.Row(point)

    cdef int _point_to_column(self, fuego_c.SgPoint point):
        """Extract the column from a point."""

        return fuego_c.Col(point) - 1

    def _assert_move_was_ok(self):
        """Assert that the most recently-played move was acceptable."""

        assert not self._board.LastMoveInfo(fuego_c.GO_MOVEFLAG_ILLEGAL)
        assert not self._board.LastMoveInfo(fuego_c.GO_MOVEFLAG_SUICIDE)
        assert not self._board.LastMoveInfo(fuego_c.GO_MOVEFLAG_REPETITION)

fuego_c.SgInit()
fuego_c.GoInit()

