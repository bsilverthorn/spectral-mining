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

    def outcomes_of(self, state, (i, j)):
        (board, player) = state

        if player == self._player:
            yield ((board.make_move(self._player, i, j), -1 * self._player), 1.0)
        else:
            assert i is None and j is None
            assert isinstance(self._opponent, specmine.rl.RandomPolicy)

            moves = []

            for oi in xrange(3):
                for oj in xrange(3):
                    if board._grid[oi, oj] == 0:
                        moves.append((oi, oj))

            p = 1.0 / len(moves)

            for (oi, oj) in moves:
                next_state = (board.make_move(-1 * self._player, oi, oj), self._player)

                yield (next_state, p)

    def check_end(self, (board, player)):
        # XXX alternatively, terminal states are simply states with no actions

        return board.check_end()

class GoDomain(object):
    states = None # should be an iterator through all possible states or just 
                  # sampled ones?

    def __init__(self, player = 1, opponent = None, size = 9):
        self._player = player
        self._opponent = opponent
        self.size = size
        self.board = specmine.go.BoardState()

#        if GoDomain.states is None:
#            GoDomain.states = specmine.go.load_adjacency_dict() # TODO - implement this method

        self.initial_state = (1,specmine.go.BoardState())

    def actions_in(self, player):
        ''' return the available actions for player for current board config '''
        if self.board.get_winner() is None:
            if player == self._player:
                for i in xrange(self.size):
                    for j in xrange(self.size):
                        if gge.gg_is_legal(player, (i,j)):
                            yield (i, j)

        yield (None, None)

    def reward_in(self, player):
        winner = self.board.get_winner()

        if winner != None:
            return winner
        else:
            return 0

    def outcome_of(self, (player,board), (i, j)):
        
        if player == self._player:
            self.board = board.make_move(player, i, j) 
            return (self._player*-1, self.board.copy())

        else:
            assert i is None and j is None

            (opponent_i, opponent_j) = self._opponent[state]
            self.board = board.make_move(player, opponent_i, opponent_j) 

            return (self._player, self.board.copy())

    def check_end(self, (player,board)):
        return board.check_end()
