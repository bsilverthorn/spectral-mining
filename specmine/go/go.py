import tarfile
import re
import glob
import fnmatch
import numpy
import specmine
import gnugo_engine as gge

logger = specmine.get_logger(__name__)

class GameState(object):
    """Full go game history."""

    def __init__(self, moves, board):
        self.moves = moves
        self.board = board

class BoardState(object):
    """Go board."""

    def __init__(self, grid = None, size = 9):
        ''' currently no way to start with nonzero grid'''

        if grid is None:
            grid = numpy.zeros((size, size))

        #self._grid = self.canonical_board(grid)
        self.grid = grid
        self._string = str(self._grid)

    def __hash__(self):
        return hash(self._string)

    def __eq__(self, other):
        return self._string == other._string

    def __str__(self):
        return self._string

    def __repr__(self):
        return repr(self.grid)

    def canonical_board(self,grid):
        grids = []
        grids.append(grid)

        for i in xrange(1,4):
            grids.append(numpy.rot90(grid,i))

        grids.append(numpy.fliplr(grid))
        grids.append(numpy.flipud(grid))

        return max(grids,key = lambda x: hash(str(x)))

    @staticmethod
    def from_gge():
        return BoardState(gge.gg_get_board())

def convert_sgf_moves(moves): 
    ''' ex: game_gen = sgf_game_reader(static_path)
            player, move = game_gen.next()'''

    move_dict = {'a':0,'b':1,'c':2,'d':3,'e':4,'f':5,'g':6,'h':7,'i':8}
    for m in moves:
        if m[0] == 'B':
            player = 1
        elif m[0] == 'W':
            player = -1
        else:
            raise RuntimeError("color error")
    
        if m[1:] == '[]': # represents a pass
            move = (-1,-1)
        else:
            move = (move_dict[m[3]],move_dict[m[2]])
        
        yield (player,move)

def sgf_game(sgf_file,min_moves=10,rating_thresh=1800):
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
            logger.warning('SGF file is not a 9x9 game')
            return None
    else: 
        logger.warning('no size given')
        return None

    # check that the handicap is zero
    # assumed zero if none given
    match = hand.findall(sgf_string)
    if len(match) > 0:
        handicap = match[0]
        if handicap != '0':
            logger.warning('SGF file has nonzero handicap')
            return None

    # check that both players have rating above the threshold 
    match1 = wrating.findall(sgf_string)
    match2 = brating.findall(sgf_string)
    if (len(match1) > 0) & (len(match2) > 0):
        white_rating = int(match1[0])
        black_rating = int(match2[0])
        if not ((white_rating>rating_thresh)&(black_rating>rating_thresh)):
            logger.warning('one of the players is below threshold skill')
            return None
    else: 
        logger.warning('one player ratings were not given')
        return None
    
    winner = result.findall(sgf_string)
    if len(winner)>0:
        winner = winner[0]
        winner = 1 if winner == 'B' else -1

    return convert_sgf_moves(moves), winner

#def check_end(self):
    #"""Is this board state an end state? ask Gnugo if it would pass on
    #both turns."""
    #move = gge.gg_genmove(1) 
    #if move == (-1,-1): # if gnugo thinks we should pass
        #move = gge.gg_genmove(-1)
        #if move == (-1,-1):
            #return True

    #return False

def read_expert_episode(sgf_file):
    gge.gg_clear_board(9)

    S = []; R = []

    out = sgf_game(sgf_file)

    if out is None:
        return S,R

    moves, winner = out

    if moves is not None:
        for (player,move) in moves:
            assert gge.gg_is_legal(player, move)

            gge.gg_play_move(player, move)

            S.append(GameState(list(moves), BoardState.from_gge()))
            R.append(0)

        R[-1] = winner

    return S, R

def estimate_value(game_state, num_rollouts):
    
    value = 0
    for i in xrange(num_rollouts):
        gge.init()
        player = 1
        for (color,move) in game_state.moves:
            gge.gg_play_move(color,move)
            player = -1*color
        
        done = False
        while !done:
            move = gge.gg_genmove(player)
            if move == (-1,-1):
                player = -1*player
                move = gge.gg_genmove(player)
                if move == (-1,-1):
                    done = True

            gge.gg_play_move(move)

        winner = gge.gg_get_winner()
        value+= winner
        
    return value/float(num_rollouts)

def main():
    games_dir = specmine.util.static_path('go_games/')
    archive_files = glob.glob(games_dir+'*.tar.bz2')
    
    for af in archive_files:
        print 'opening archive: ', af
        archive = tarfile.open(af)
        names = archive.getnames()

        for name in names:
            if not fnmatch.fnmatch(name, '*/*/*/*.sgf'):
                continue
            print 'playing game: ', name
            f = archive.extractfile(name)
            s,r = read_expert_episode(f)
            print s,r

            f.close()

if __name__ == "__main__":
    main()

