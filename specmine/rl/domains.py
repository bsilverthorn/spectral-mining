import numpy
import specmine

# XXX other code assumes that actions in states with multiple actions are deterministic

class SimpleRoomDomain(object):
    def __init__(self, width = 8):
        self._width = width
        self._actions = map(numpy.array, [[1, 0], [-1, 0], [0, 1], [0, -1], [0, 0]])

    def actions_in(self, (state_x, state_y)):
        for (action_x, action_y) in self._actions:
            if 0 <= state_x + action_x < self._width and 0 <= state_y + action_y < self._width:
                yield (action_x, action_y)

    def reward_in(self, (x, y)):
        if x == 0 and y == 0:
            return 0
        else:
            return -1

    def outcome_of(self, (state_x, state_y), (action_x, action_y)):
        return (state_x + action_x, state_y + action_y)

    @property
    def states(self):
        for x in xrange(self._width):
            for y in xrange(self._width):
                yield (x, y)

class TicTacToeDomain(object):
    states = None

    def __init__(self, player = 1, opponent = None):
        self._player = player
        self._opponent = opponent

        if TicTacToeDomain.states is None:
            TicTacToeDomain.states = specmine.tictac.load_adjacency_dict()

        self.initial_state = (specmine.tictac.BoardState(), 1)

    def actions_in(self, (board, player)):
        if board.get_winner() is None:
            if player == self._player:
                for i in xrange(3):
                    for j in xrange(3):
                        if board.grid[i, j] == 0:
                            yield (i, j)
            else:
                yield (None, None)

    def reward_in(self, (board, player)):
        winner = board.get_winner()

        if winner != None:
            return winner
        else:
            return 0

    def outcome_of(self, state, (i, j)):
        (board, player) = state

        if player == self._player:
            return (board.make_move(self._player, i, j), -1 * self._player)
        else:
            assert i is None and j is None

            (opponent_i, opponent_j) = self._opponent[state]

            return (board.make_move(-1 * self._player, opponent_i, opponent_j), self._player)

    def check_end(self, (board, player)):
        # XXX alternatively, terminal states are simply states with no actions

        return board.check_end()

