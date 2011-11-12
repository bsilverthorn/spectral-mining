import cPickle as pickle
import numpy
import specmine

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
    # XXX assumes player 1

    def __init__(self, opponent = None):
        # construct opponent
        if opponent is None: 
            pickle_path = specmine.util.static_path("ttt_optimal.pickle.gz")

            with specmine.util.openz(pickle_path) as pickle_file:
                optimal = pickle.load(pickle_file)

                class OptimalOpponent(object):
                    def __getitem__(self, board):
                        return optimal[board][:2]

                self._opponent = OptimalOpponent()
        else:
            self._opponent = opponent

        # construct state space
        pickle_path = specmine.util.static_path("ttt_states.pickle.gz")

        with specmine.util.openz(pickle_path) as pickle_file:
            states = pickle.load(pickle_file)

        self.states = states
        self.initial_state = (specmine.tictac.BoardState(),1)

    def actions_in(self, (board, player)):
        if board.get_winner() is None:
            if player == 1:
                for i in xrange(3):
                    for j in xrange(3):
                        if board.grid[i, j] == 0:
                            yield (i, j)
            else:
                yield (None, None)

    def reward_in(self, (board,player)):
        winner = board.get_winner()
        if winner != None:
            return winner
        else:
            return 0

    def outcome_of(self, (board,player), (i, j)):
        if player == 1:
            return (board.make_move(1, i, j), -1)
        else:
            assert player == -1
            assert i is None and j is None

            (opponent_i, opponent_j) = self._opponent[board]

            return (board.make_move(-1, opponent_i, opponent_j), 1)

    def check_end(self, (board,player)):
        return board.check_end()
