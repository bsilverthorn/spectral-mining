import copy
import numpy

cimport cython
cimport numpy

cdef extern from "gnugo-3.8/engine/clock.h":
    void set_level(int new_level) # function for setting the level (difficulty/search depth) of the gnugo engine
    int get_level()
    
cdef extern from "gnugo-3.8/engine/board.h":

    ctypedef unsigned char Intersection
    Intersection board[] #   
    int black_captured   #     
    int white_captured 
    float        komi
    int          handicap   # /* used internally in chinese scoring */

    #/* Functions handling the permanent board state. */
    
    void add_stone(int pos, int color)
    void remove_stone(int pos)
    void play_move(int pos, int color)
    int undo_move(int n)
    int is_legal(int pos, int color)

cdef extern from "gnugo-3.8/sgf/sgftree.h":

    cdef struct SGFProperty_t:
        SGFProperty_t* next
        short name
        char* value
    ctypedef  SGFProperty_t SGFProperty

    cdef struct SGFNode_t:
        SGFProperty* props
        SGFNode_t* parent
        SGFNode_t* child
        SGFNode_t* next
    ctypedef SGFNode_t SGFNode

    cdef struct SGFTree_t:
        SGFNode* root
        SGFNode* lastnode
    ctypedef SGFTree_t SGFTree #optional  
     
cdef extern from "gnugo-3.8/utils/gg_utils.h":
    pass
cdef extern from "gnugo-3.8/engine/liberty.h":
    pass
    
cdef extern from "gnugo-3.8/engine/gnugo.h":

    void init_gnugo(float memory, unsigned int random_seed)
    void gnugo_clear_board(int boardsize)
    int gnugo_sethand(int desired_handicap, SGFNode *node) # not implemented (including debugging imports from sgftree)
    void showboard(int xo)  #/* ascii rep. of board to stderr */
    
    ctypedef struct Gameinfo:
        int handicap
        int to_move		#/* whose move it currently is */
        SGFTree game_record	#/* Game record in sgf format. */
        int computer_player	#/* BLACK, WHITE, or EMPTY (used as BOTH) */
    
    int genmove(int color, float* value, int* resign)
    int genmove_conservative(int color, float *value)
    float gnugo_estimate_score(float *upper, float *lower)
    float aftermath_compute_score(int color, SGFTree *tree)

global board

#
# GNUGO WRAPPERS
#

def gg_white_captured():
    return white_captured
    
def gg_black_captured():
    return black_captured

def gg_clear_board(boardsize):
    ''' clear go board, also used to set board size '''
    gnugo_clear_board(boardsize)

def gg_init(memory = -1, seed = 0):
    """
    Initialize the gnugo engine.
    
    Must be called at least once before any gnugo methods are used.
    """

    init_gnugo(memory,seed)

def gg_get_level():
    ''' get current gnugo engine level/difficulty '''

    return get_level()

def gg_set_level(int level):
    """Set the difficulty level of the gnugo engine."""

    assert level > 0
    assert level <= 10

    set_level(level)

def gg_showboard():
    ''' output ascii representation of board to stdout '''
    showboard(1)

def gg_get_board():
    ''' return board_size by board_size numpy array representation of current board '''
    # XXX
    b = []
    for i in range(11,101):
        b.append(board[i]) # creates the list from the vector
    b = numpy.array(b, int)
    b = b.reshape(9,9+1)
    b = b[:,:-1]
    c = copy.deepcopy(b)
    c[(b==2).nonzero()] = 1
    c[(b==1).nonzero()] = -1
    return c

def gg_get_inverse_board():
    ''' get the current board with all black stone white and white stones black '''
    b = gg_get_board()
    c = copy.deepcopy(b)
    c[(b==-1).nonzero()] = 1
    c[(b==1).nonzero()] = -1
    return c

def gg_genmove(player):
    """Return the best move according to gnugo."""

    player = 1 if player == -1 else 2
    #cdef float* v = None
    #cdef int* r = None
    move = genmove(player, NULL, NULL)
    #move = genmove_conservative(player, v)

    return pos2xy(move)

def gg_play_move(player, move_x, move_y):
    """Update game state to reflect specified move."""

    player = 1 if player == -1 else 2

    play_move(xy2pos(move_x, move_y), player)

def gg_undo_move(int num):
    ''' undo the last num moves '''
    sucess = undo_move(num)
    return sucess

def gg_add_stone( int player, move):
    ''' add stone at position move, different than play_move in that it just changes
    the board state, and doesn't have to follow the rules of the game (I think) '''
    player = 1 if player == -1 else 2
    add_stone(xy2pos(move[0], move[1]), player)

def gg_remove_stone(move):
    ''' removes stone at position pos '''
    remove_stone(xy2pos(move[0], move[1]))

def gg_is_legal(int player, int move_i, int move_j):
    ''' checks if move is legal (not sure if this includes suicides) '''

    player = 1 if player == -1 else 2

    return is_legal(xy2pos(move_i, move_j), player)

def gg_estimate_score():
    ''' get gnugo's current estimate of the board score along with upper and lower bounds
    with positive being good for for black (which is the opposite convention as used by gnugo) '''
    cdef float upper
    cdef float lower
    cdef float* up = &upper
    cdef float* low = &lower
    score = gnugo_estimate_score(up, low);

    return (-score, -upper, -lower)

def gg_aftermath_score(int player):
    ''' compute the aftermath score if the game were to be played out to completion
    (for use after both players have passed) '''
    player = 1 if player == -1 else 2
    cdef SGFTree tree
    cdef SGFTree* tree_p = &tree
    after_score = aftermath_compute_score(player, tree_p)
    return after_score

def gg_get_winner_assumed():
    """Return the winner of the game."""

    (score, up, low) = gg_estimate_score()

    if score > 0:
        return 1
    elif score == 0:
        return 0
    else:
        return -1

def gg_get_winner():
    """Return the winner of the game, if any."""

    move1 = gg_genmove(1)        
    move2 = gg_genmove(-1)

    if move1 == move2 == (-1,-1):
        (score, up, low) = gg_estimate_score()

        if score > 0:
            return 1
        else:
            return -1

    return None

#
# ADDED FUNCTIONALITY
#

cdef int xy2pos(int x, int y):
    """
    Convert from xy position to linear gnugo board position.

    Assumes board_size is 9 and MAX_BOARD in gnugo.h is set to 9.
    """

    return (9 + 2) + x * (9 + 1) + y

cdef object pos2xy(int pos):
    """Convert a linear gnugo board position to an xy position."""

    return (pos / (9 + 1) - 1, (pos % (9 + 1)) - 1)

@cython.infer_types(True)
def replay_game(moves):
    """Replay a game, returning an array of grids."""

    cdef int M = moves.shape[0]

    cdef numpy.ndarray[numpy.int8_t, ndim = 2] moves_M3 = moves
    cdef numpy.ndarray[numpy.int8_t, ndim = 3] grids_M99 = numpy.empty((M, 9, 9), numpy.int8)

    gg_clear_board(9)

    for m in xrange(M):
        player = moves_M3[m, 0]
        move_x = moves_M3[m, 1]
        move_y = moves_M3[m, 2]

        if not gg_is_legal(player, move_x, move_y):
            raise ValueError("illegal move: {0} at ({1}, {2})".format(player, move_x, move_y))

        gg_play_move(player, move_x, move_y)

        for x in xrange(9):
            for y in xrange(9):
                piece = board[xy2pos(x, y)]

                if piece == 1:
                    grids_M99[m, x, y] = -1
                elif piece == 2:
                    grids_M99[m, x, y] = 1
                else:
                    grids_M99[m, x, y] = piece

    return grids_M99

def legal_moves(player):
    """Iterate over legal moves."""

    for i in xrange(9):
        for j in xrange(9):
            if gg_is_legal(player, i, j):
                yield (i, j)

#def check_end(self):
    #"""Is this board state an end state? ask Gnugo if it would pass on
    #both turns."""
    #move = gge.gg_genmove(1) 
    #if move == (-1,-1): # if gnugo thinks we should pass
        #move = gge.gg_genmove(-1)
        #if move == (-1,-1):
            #return True

    #return False

gg_init()

