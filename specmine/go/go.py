import tarfile
import re
import glob
import fnmatch
import numpy
import specmine
import gnugo_engine as gge


class BoardState(object):
    """Go board."""

    def __init__(self, size = 9):
        ''' currently no way to start with nonzero grid'''
        grid = numpy.zeros((size, size))
        self._grid = grid
        self._string = str(self._grid)
        gge.gg_init()

    def __hash__(self):
        return hash(self._string)

    def __eq__(self, other):
        return self._string == other._string

    def __str__(self):
        return self._string

    def __repr__(self):
        return repr(self._grid)

    def make_move(self, player, i, j):
    #mutable - returns this rather than making new object
        assert gge.gg_is_legal(player,(i,j))
        
        gge.gg_play_move(player,(i,j))
        print 'get board: ', gge.gg_get_board()
        print type(gge.gg_get_board())
        self._grid = self.canonical_board(gge.gg_get_board())
        self._string = str(self._grid)
        
        return self

    def get_winner(self):
        """Return the winner of the game, if any."""
        move1 = gge.gg_genmove(1)        
        move2 = gge.gg_genmove(-1)
        if move1 == move2 == (-1,-1):
            (score,up,low) = gge.gg_get_score()
            if score > 0:
                return 1
            else:
                return -1

        return None

    def check_end(self):
        """Is this board state an end state? ask Gnugo if it would pass on
        both turns."""
        move = gge.gg_genmove(1)
        if move == (-1,-1): # if gnugo thinks we should pass
            move = gge.gg_genmove(-1)
            if move == (-1,-1):
                return True

        return False

    @property
    def grid(self):
        return self._grid

    def canonical_board(self,grid):

        grids = [].append(grid)
        for i in xrange(1,4):
            grids.append(numpy.rot90(grid,i))
        grids.append(numpy.fliplr(grid))
        grids.append(numpy.flipud(grid))

        return max(grids,key = lambda x: hash(str(x)))

def generate_sgf(sgf_string):
    
    ''' ex: game_gen = sgf_game_reader(static_path)
            player, move = game_gen.next()'''

    move_dict = {'a':0,'b':1,'c':2,'d':3,'e':4,'f':5,'g':6,'h':7,'i':8}
    move = re.compile("[BW]\[[a-z]{0,2}\]") 
    
    # parse all moves
    moves = move.findall(sgf_string)
    for m in moves:
        if m[0] == 'B':
            player = 1
        elif m[0] == 'W':
            player = -1
        else:
            print 'color error'
    
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

    
    sgf_string = sgf_file.read()  
    # check board is the right size
    match = size.findall(sgf_string)
    if len(match) > 0:
        board_size = match[0]
        if board_size != '9':
            print 'SGF file is not a 9x9 game'
            return None
    else: 
        print 'no size given'
        return None
    
    # check that the handicap is zero
    # assumed zero if none given
    match = hand.findall(sgf_string)
    if len(match) > 0:
        handicap = match[0]
        if handicap != '0':
            print 'SGF file has nonzero handicap'
            return None

    # check that both players have rating above the threshold 
    match1 = wrating.findall(sgf_string)
    match2 = brating.findall(sgf_string)
    if (len(match1) > 0) & (len(match2) > 0):
        white_rating = int(match1[0])
        black_rating = int(match2[0])
        if not ((white_rating>rating_thresh)&(black_rating>rating_thresh)):
            print 'one of the players is below threshold skill'
            return None
    else: 
        print 'one player ratings were not given'
        return None
    
    winner = result.findall(sgf_string)
    if len(winner)>0:
        winner = winner[0]
        winner = 1 if winner == 'B' else -1

    return generate_sgf(sgf_string), winner


def generate_expert_episode(sgf_file):
    S = []; R = []
    board = BoardState()
    
    out = sgf_game(sgf_file)
    if out == None:
        return S,R
    moves,winner = out
    moves = list(moves)
    print 'moves: ', moves
    print 'winner: ',winner
    if moves != None:
        for (player,move) in moves:
            board = board.make_move(player,move[0],move[1])
            S.append(board)
            R.append(0)
        
        print len(R)
        print R
        R[-1] = winner

    return S,R

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

            f = archive.extractfile(name)
            s,r = generate_expert_episode(f)
            print s,r

            f.close()

if __name__ == "__main__":
    main()

