import plac
import play_tictac

if __name__ == "__main__":
    plac.call(play_tictac.main)

from random import choice
import gzip
import cPickle as pickle
import contextlib
import numpy
import tictac

def ttt_value_min(board, player, alpha = -numpy.inf, beta = numpy.inf):
    """Compute the state value (min node) with alpha-beta pruning."""

    # terminal state?
    winner = board.get_winner()

    if winner is not None:
        return -1 * player * winner

    # no; recurse
    min_value = numpy.inf

    for i in xrange(3):
        for j in xrange(3):
            if board._grid[i, j] == 0:
                value = ttt_value_max(board.make_move(player, i, j), -1 * player, alpha, beta)

                if value <= alpha:
                    return value

                if min_value > value:
                    min_value = value

                    if beta > value:
                        beta = value

    return min_value

def ttt_value_max(board, player, alpha = -numpy.inf, beta = numpy.inf):
    """Compute the state value (max node) with alpha-beta pruning."""

    # terminal state?
    winner = board.get_winner()

    if winner is not None:
        return player * winner

    # no; recurse
    max_value = -numpy.inf

    for i in xrange(3):
        for j in xrange(3):
            if board._grid[i, j] == 0:
                value = ttt_value_min(board.make_move(player, i, j), -1 * player, alpha, beta)

                if value >= beta:
                    return value

                if max_value < value:
                    max_value = value

                    if alpha < value:
                        alpha = value

    return max_value

def ttt_optimal_move(board, player = 1):
    """Compute the optimal move in a given state."""

    max_i = None
    max_j = None
    max_value = -numpy.inf

    for i in xrange(3):
        for j in xrange(3):
            if board._grid[i, j] == 0:
                value = ttt_value_min(board.make_move(player, i, j), -1 * player)

                if max_value < value:
                    max_i = i
                    max_j = j
                    max_value = value

    return (max_i, max_j, max_value)

def test_ttt_minimax():
    configurations = [
        [[-1,  0,  0],
         [ 0, -1,  1],
         [ 1,  0,  0]],
        [[ 1, -1, -1],
         [-1,  1,  1],
         [ 1,  0, -1]],
        [[-1,  1, -1],
         [-1,  1,  1],
         [ 0,  0, -1]],
        [[ 0,  0,  1],
         [-1, -1, -1],
         [ 0,  1,  0]],
        ]
    boards = [tictac.BoardState(numpy.array(c)) for c in configurations]

    assert ttt_optimal_move(boards[0], 1) == (2, 2, 1)
    assert ttt_optimal_move(boards[1], 1) == (2, 1, 0)
    assert ttt_optimal_move(boards[2], 1) == (2, 1, 1)
    assert ttt_optimal_move(boards[2], -1) == (2, 0, 1)
    assert ttt_value_max(boards[3], -1) == 1

def rl_agent_choose_move(board,player,index,v):
    '''returns the next move chosen according to the value function v
    along with the corresponding board state index'''

    valid_moves = numpy.array(board._grid.nonzero())
    next_move_indxs = []
    for m in xrange(valid_moves.shape[1]):
        after_state = board.make_move(player,valid_moves[0,m],valid_moves[1,m])
        next_move_indxs.append(index[after_state])

    # choose randomly among the moves of highest value
    max_value = numpy.max(v[next_move_indxs])
    best_moves = next_move_indxs[v[next_move_indxs] == max_value]
    return choice(best_moves)

def play_against_optimal(v, index, rindex):
    S = [] # list of board state indices
    R = [] # list of rewards len(R) = (len(S)-1)
    board = tictac.BoardState(numpy.zeros(3,3));
    S.append(index[board])
    
    # choose first player randomly
    player = numpy.round(numpy.random.rand)*2-1 

    while board.get_winner() == None:
        # rl agent is always player 1
        if player == 1:
            move_indx = rl_agent_choose_move(board,player,index,v)
            move = rindex[move_indx]
            board = board.make_move(player,move[0],move[1])
            R.append(0)
            S.append(move_indx)
                    
        else:
            move = ttt_optimal_move(board,player)
            board = board.make_move(player,move[0],move[1])

            
            
        
    
    
    

def train_rl_ttt_agent(k=50,num_games=10):
    phi, index = tictac.get_ttt_laplacian_basis(k)
     


@plac.annotations(
    out_path = ("path to write policy pickle",),
    states_path = ("path to TTT states pickle",),
    )
def main(out_path, states_path = "ttt_states.pickle.gz"):
    """Generate the optimal TTT policy."""

    with contextlib.closing(gzip.GzipFile(states_path)) as pickle_file:
        states = pickle.load(pickle_file)

    policy = {}

    for (i, state) in enumerate(states):
        move = ttt_optimal_move(state)

        policy[state] = move

        print "optimal move in state #{0}: {1}".format(i, move[:2])

    with contextlib.closing(gzip.GzipFile(out_path, "w")) as pickle_file:
        pickle.dump(policy, pickle_file)

