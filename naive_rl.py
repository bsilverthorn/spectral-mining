import gzip
import cPickle as pickle
import contextlib
import numpy
import tictac

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
    def __init__(self, opponent = None):
        if opponent is None:
            with contextlib.closing(gzip.GzipFile("ttt_optimal.pickle.gz")) as pickle_file:
                # XXX not quite a "policy" according to our interface
                self._opponent = pickle.load(pickle_file)
        else:
            self._opponent = opponent

        with contextlib.closing(gzip.GzipFile("ttt_states.pickle.gz")) as pickle_file:
            self._boards = pickle.load(pickle_file)

    def actions_in(self, (player, board)):
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
            return (-1 * player, board.make_move(player, i, j))
        else:
            assert i == None and j == None

            (opponent_i, opponent_j, _) = self._opponent[board]

            return (-1 * player, board.make_move(player, opponent_i, opponent_j))

    @property
    def states(self):
        for board in self._boards:
            yield (1, board)
            yield (-1, board)

class QFunctionPolicy(object):
    def __init__(self, domain, q_values):
        self._domain = domain
        self._q_values = q_values

    def __getitem__(self, state):
        return max(self._domain.actions_in(state), key = self._q_values.__getitem__)

def learn_q_values(domain, rate = 1e-1, discount = 9e-1, iterations = 1000):
    q_values = {}

    for state in domain.states:
        for action in domain.actions_in(state):
            q_values[(state, action)] = 0.0

    for _ in xrange(iterations):
        for state in domain.states:
            for action in domain.actions_in(state):
                # compute the action value
                next_state = domain.outcome_of(state, action)
                max_next_value = max(q_values[(next_state, a)] for a in domain.actions_in(next_state))

                # update our table
                value = q_values[(state, action)]
                error = domain.reward_in(next_state) + discount * max_next_value - value

                q_values[(state, action)] = value + rate * error

    return q_values

def learn_q_policy(domain, **options):
    q_values = learn_q_values(domain, **options)

    return QFunctionPolicy(domain, q_values)

def main():
    # build the state space
    domain = TicTacToeDomain()
    policy = learn_q_policy(domain)

if __name__ == "__main__":
    main()

