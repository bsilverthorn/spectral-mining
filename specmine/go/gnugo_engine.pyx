import numpy as np
import copy as cp

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


def gg_white_captured():
    return white_captured
    
def gg_black_captured():
    return black_captured

def gg_clear_board(boardsize):
    ''' clear go board, also used to set board size '''
    gnugo_clear_board(boardsize)

def gg_init(memory = -1, seed = 0):
    ''' initialize the gnugo engine, must be called once at the beginning of use'''
    init_gnugo(memory,seed)

def gg_get_level():
    ''' get current gnugo engine level/difficulty '''
    level = get_level()
    return level

def gg_set_level(int level):
    ''' set gnugo engine level/difficulty '''
    if (level > 0) & (level <= 10):
        set_level(level)
        return True
    else:
        return False

def gg_showboard():
    ''' output ascii representation of board to stdout '''
    showboard(1)

def gg_get_board():
    ''' return board_size by board_size numpy array representation of current board '''
    b = []
    for i in range(11,101):
        b.append(board[i]) # creates the list from the vector
    b = np.array(b, int)
    b = b.reshape(9,9+1)
    b = b[:,:-1]
    return b

def gg_get_inverse_board():
    ''' get the current board with all black stone white and white stones black '''
    b = gg_get_board()
    c = cp.deepcopy(b)
    c[(b==3).nonzero()] = 2
    c[(b==2).nonzero()] = 3
    return c

def gg_genmove(player):
    ''' have gnuggo generate best move.
    Used primarily to know when to pass to avoid overly long games  '''
    player = 1 if player == -1 else 2
    cdef float* v
    cdef int* r
    move = genmove(player, v, r)
    #move = genmove_conservative(player, v)
    x = (move) / (9 + 1) - 1
    y = (move % (9 + 1)) - 1
    return (x,y)
    
def gg_play_move(player,move):
    ''' play move at position (move[0],move[1]) (row,col notation), record in game state '''
    player = 1 if player == -1 else 2
    pos = ((9 + 2) + move[0] * (9 + 1) + move[1])
    play_move(pos, player)

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

def gg_is_legal(int player, move):
    ''' checks if move is legal (not sure if this includes suicides) '''
    player = 1 if player == -1 else 2
    legal = is_legal(xy2pos(move[0], move[1]), player)
    return legal

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
    
cdef int xy2pos(int x, int y):
    ''' convert from xy position to linear gnugo board position assuming board_size is 9 and MAX_BOARD in gnugo.h is set to 9'''
    return ((9 + 2) + x * (9 + 1) + y)
