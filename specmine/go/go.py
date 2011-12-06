import re
import random
import hashlib
import numpy
import specmine
import gnugo_engine as gge

logger = specmine.get_logger(__name__)

class Game(object):
    """A single Go game in its entirety."""

    def __init__(self, moves, grids, winner):
        """
        Initialize.

        moves: Mx3 array of (player, i, j) rows;
        grids: Mx9x9 array of stone grids
        """

        self.moves = moves
        self.grids = grids
        self.winner = winner

    def get_state(self, m):
        """Get the game state after a particular move."""

        board = BoardState(self.grids[m])

        return GameState(self.moves[:m + 1], board)

class GameState(object):
    """Instantaneous state of a Go game."""

    def __init__(self, moves, board):
        self.moves = numpy.asarray(moves, numpy.int8)
        self.board = board

    # hash and equality only rely on the board state, not the list of moves (?)
    def __hash__(self):
        return hash(self.board._string)

    def __eq__(self,other):
        return self.board._string == other.board._string

class BoardState(object):
    """Go board."""

    def __init__(self, grid = None, key = None):
        if grid is None:
            grid = numpy.zeros((9, 9), numpy.int8)
        else:
            grid = numpy.asarray(grid, numpy.int8)

        if key is None:
            key = hashlib.md5(grid).digest()

        self.grid = grid
        self.key = key

    def __hash__(self):
        return hash(self.key)

    def __eq__(self, other):
        return self.key == other.key

    def __str__(self):
        return str(self.grid)

    def __repr__(self):
        return repr(self.grid)

    #def canonical(self):
        #grids = []
        #grids.append(grid)

        #for i in xrange(1,4):
            #grids.append(numpy.rot90(grid,i))

        #grids.append(numpy.fliplr(grid))
        #grids.append(numpy.flipud(grid))
        #grids.append(numpy.fliplr(numpy.rot90(grid,1)))
        #grids.append(numpy.flipud(numpy.rot90(grid,1)))

        #return BoardState(max(grids, key = lambda x: hash(str(x))))

    @staticmethod
    def from_gge():
        return BoardState(gge.gg_get_board())

def convert_sgf_moves(moves): 
    ord_a = ord("a")

    for m in moves:
        if m[0] == 'B':
            player = 1
        elif m[0] == 'W':
            player = -1
        else:
            raise RuntimeError("unrecognized player color")

        if m[1:] == '[]':
            # player passed
            yield (player, -1, -1)
        else:
            # player moved
            yield (player, ord(m[3]) - ord_a, ord(m[2]) - ord_a)

def read_sgf_game_moves(sgf_file, min_moves=10, rating_thresh=1800):
    ''' check if sgf has enough moves and minimum rating, etc 
    return move generator if so'''

    size = re.compile("SZ\[([0-9]+)\]") # check board size
    hand = re.compile("HA\[([0-9]+)\]") # and handicap
    wrating = re.compile("WR\[([0-9]+)\]")
    brating = re.compile("BR\[([0-9]+)\]")
    result = re.compile("RE\[([W|B]).*\]")
    move = re.compile("[BW]\[[a-z]{0,2}\]") 

    sgf_string = sgf_file.read() 

    # parse all moves
    moves = move.findall(sgf_string)

    if len(moves) < min_moves:
        return None
 
    # check board is the right size
    match = size.findall(sgf_string)

    if len(match) > 0:
        board_size = match[0]

        if board_size != '9':
            logger.debug('SGF file is not a 9x9 game')

            return None
    else: 
        logger.debug('SGF file does not specify size')

        return None

    # check that the handicap is zero (assumed zero if none given)
    match = hand.findall(sgf_string)

    if len(match) > 0:
        handicap = match[0]

        if handicap != '0':
            logger.debug('SGF file has nonzero handicap')

            return None

    # check that both players have rating above the threshold 
    match1 = wrating.findall(sgf_string)
    match2 = brating.findall(sgf_string)

    if (len(match1) > 0) & (len(match2) > 0):
        white_rating = int(match1[0])
        black_rating = int(match2[0])

        if not ((white_rating>rating_thresh)&(black_rating>rating_thresh)):
            logger.debug('one of the players is below threshold skill')

            return None
    else: 
        logger.debug('one player ratings were not given')

        return None
    
    winner = result.findall(sgf_string)

    if len(winner)>0:
        winner = winner[0]
        winner = 1 if winner == 'B' else -1

    return (list(convert_sgf_moves(moves)), winner)

def read_sgf_game(sgf_file):
    """
    Read a Go game from an SGF file.

    Returns an instance of specmine.go.Game.
    """

    game_raw = read_sgf_game_moves(sgf_file)

    if game_raw is None:
        return None

    (raw_moves, winner) = game_raw

    moves = numpy.array(raw_moves, dtype = numpy.int8)
    grids = gge.replay_game(moves)

    return Game(moves, grids, winner)

def estimate_value(game_state, rollouts = 32, epsilon = 0.2):
    value = 0.0

    gge.gg_set_level(1)

    for i in xrange(rollouts):
        gge.gg_clear_board(9)

        player = -1
        
        # get to current board state
        for (player, move_x, move_y) in game_state.moves:
            if move_x != -1:
                gge.gg_play_move(player, move_x, move_y)

        passed = 0

        while passed < 2:
            player *= -1

            #print BoardState.from_gge().grid

            if numpy.random.rand() >= epsilon:
                (move_x, move_y) = gge.gg_genmove(player)
            else:
                options = list(gge.legal_moves(player))

                if len(options) > 0:
                    (move_x, move_y) = random.choice(options)
                else:
                    move_x = -1

            #print move_x, move_y

            if move_x == -1:
                passed += 1

                continue
            else:
                passed = 0

                gge.gg_play_move(player, move_x, move_y)

        winner = gge.gg_get_winner_assumed()
        value += winner


        logger.debug("player %i won rollout %i of %i", winner, i + 1, rollouts)

    gge.gg_set_level(10) 

    return value / float(rollouts)

def load_affinity_graph(path=None):
    """loads the preconstructed affinity graph for 9x9 Go"""

    states_path = specmine.util.static_path("ttt_states.pickle.gz")

    logger.info("loading TTT adjacency dict from %s", states_path)

    with specmine.util.openz(states_path) as pickle_file:
        return pickle.load(pickle_file

