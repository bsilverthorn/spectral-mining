import gzip
import cPickle as pickle
import contextlib
import numpy
import plac

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
            with contextlib.closing(gzip.GzipFile("ttt_optimal.pickle.gz")) as pickle_file:
                optimal = pickle.load(pickle_file)

                class OptimalOpponent(object):
                    def __getitem__(self, board):
                        return optimal[board][:2]

                self._opponent = OptimalOpponent()
        else:
            self._opponent = opponent

        # construct state space
        with contextlib.closing(gzip.GzipFile("ttt_states.pickle.gz")) as pickle_file:
            boards = pickle.load(pickle_file)

        self._states = [(board.get_player(), board) for board in boards]

    def actions_in(self, (player, board)):
        if board.get_winner() is None:
            if player == 1:
                for i in xrange(3):
                    for j in xrange(3):
                        if board.grid[i, j] == 0:
                            yield (i, j)
            else:
                yield (None, None)

    def reward_in(self, (player, board)):
        if board.get_winner() == 1:
            return 1
        else:
            return 0

    def outcome_of(self, (player, board), (i, j)):
        if player == 1:
            return (-1, board.make_move(1, i, j))
        else:
            assert player == -1
            assert i is None and j is None

            (opponent_i, opponent_j) = self._opponent[board]

            return (1, board.make_move(-1, opponent_i, opponent_j))

    @property
    def states(self):
        return self._states

class QFunctionPolicy(object):
    def __init__(self, domain, q_values):
        self._domain = domain
        self._q_values = q_values

    def __getitem__(self, state):
        return max(self._domain.actions_in(state), key = self._q_values.__getitem__)

def learn_q_values(domain, rate = 1e-1, discount = 9e-1, iterations = 128):
    q_values = {}

    for state in domain.states:
        for action in domain.actions_in(state):
            q_values[(state, action)] = 0.0

    for i in xrange(iterations):
        for state in domain.states:
            for action in domain.actions_in(state):
                # compute the action value
                next_state = domain.outcome_of(state, action)
                max_next_value = None

                for next_action in domain.actions_in(next_state):
                    next_value = q_values[(next_state, next_action)]

                    if max_next_value is None or max_next_value < next_value:
                        max_next_value = next_value

                # update our table
                value = q_values[(state, action)]

                if max_next_value is None:
                    error = domain.reward_in(next_state) - value
                else:
                    error = domain.reward_in(next_state) + discount * max_next_value - value

                q_values[(state, action)] = value + rate * error

                print ",".join(map(str, [i, error]))

    return q_values

def learn_q_policy(domain, **options):
    q_values = learn_q_values(domain, **options)

    return QFunctionPolicy(domain, q_values)

def main():
    # build the state space
    domain = TicTacToeDomain()
    policy = learn_q_policy(domain)

if __name__ == "__main__":
    plac.call(main)

