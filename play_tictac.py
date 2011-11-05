import plac
import play_tictac

if __name__ == "__main__":
    plac.call(play_tictac.main)

import gzip
import cPickle as pickle
import contextlib
import numpy
#import matplotlib.pyplot as plt
import tictac
import rl

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


def test_value_function(v,index,states):

    configurations = [
       [[-1.,  1.,  0.],
        [ 0., -1.,  1.],
        [ 1.,  0., -1.]],
       [[ 1., -1., -1.],
        [-1.,  1.,  1.],
        [ 1., -1.,  1.]],
       [[ 1., -1.,  1.],
        [ 1., -1., -1.],
        [ 0.,  0.,  1.]],
       ]

    boards = [tictac.BoardState(numpy.array(c)) for c in configurations]
    
    print 'negative value?: ', v[index[boards[0]]] 
    #assert v[index[boards[0]]] < 0
    print boards[1]._grid
    print 'positive value?: ', v[index[boards[1]]]
    #assert v[index[boards[0]]] < 0
    print boards[2]._grid
    print 'value (should be positive if next to move): ', v[index[boards[2]]]
   

def eps_optimal_move(board, opt_strat, eps=0):
    
    if numpy.random.random() > eps:
        return opt_strat[board][0:2]
    
    else:
        valid_moves = numpy.array(numpy.nonzero(board._grid==0))
        r = numpy.round(numpy.random.random()*(valid_moves.shape[1]-1))
        return (valid_moves[0,r],valid_moves[1,r])
        
def rl_agent_choose_move(board,index,v,eps=0):
    '''returns the next move chosen according to the value function v
    along with the corresponding board state index'''

    valid_moves = numpy.array(numpy.nonzero(board._grid==0))
    
    if numpy.random.random() > eps:
        max_value = None
        for m in xrange(valid_moves.shape[1]):

            # always plays player 1's move - invert board for player -1
            move = (valid_moves[0,m],valid_moves[1,m])
            after_state = board.make_move(1,move[0],move[1])
            move_index = index[after_state]
            value = v[move_index]

            if value > max_value:
                best_move_indxs = [move_index]
                best_moves = [move]
                max_value = value
            elif value == max_value:
                best_move_indxs.append(move_index)
                best_moves.append(move)            

        # choose randomly among the moves of highest value
        r = int(numpy.round(numpy.random.random()*(len(best_move_indxs)-1)))
        return  best_moves[r],best_move_indxs[r]

    #choose random move
    else: 
        r = int(numpy.round(numpy.random.random()*(valid_moves.shape[1]-1)))
        move = (valid_moves[0,r],valid_moves[1,r])
        after_state = board.make_move(1,move[0],move[1])
        move_index = index[after_state]

        return move, move_index


def play_opponent(v, index, opt_strat, eps_opt, eps_greedy, self_play=False): 

    S = [] # list of board state indices
    board = tictac.BoardState(numpy.zeros((3,3)));
    S.append(index[board]) 

    player = 1 # only games starting with player 1 are stored in index

    winner = None
    while winner == None:
        # rl agent is always player 1
        if player == 1:
            move, next_indx = rl_agent_choose_move(board,index,v,eps_greedy)
            board = board.make_move(player,move[0],move[1])
                    
        else:
            if self_play:
                inverse_board = tictac.BoardState(-board._grid)
                move, null = rl_agent_choose_move(inverse_board,index,v,eps_greedy)
                board = board.make_move(player,move[0],move[1])
                next_indx = index[board]
            else:
                move = eps_optimal_move(board,opt_strat,eps_opt)
                board = board.make_move(player,move[0],move[1])
                next_indx = index[board]

        player = -1*player
        
        S.append(next_indx)
        winner = board.get_winner()
        #print board._grid
    
    R = [0]*(len(S)); R[-1] = winner
    
    return S,R

def invert_episode(S,R,index,rindex):
    # TODO incomplete
    for s in S:
        board = rindex[s]
        inv_board = tictac.BoardState(-board._grid)
             

def train_rl_ttt_agent(k=50,num_games=1000,freq=100,eps_opt=0,eps_greedy=0):
    phi, index = tictac.get_ttt_laplacian_basis(k)
    n = phi.shape[0] # number of states
    beta = numpy.zeros(k) # numpy.random.standard_normal(k);
    v = numpy.dot(phi,beta)
    
    print 'learning against opponent'

    # load optimal strategy
    with gzip.GzipFile("ttt_optimal.pickle.gz") as pickle_file:
        opt_strat = pickle.load(pickle_file)
    
    # for recording performance
    win_rate = [0]*(num_games/freq)
    lose_rate = [0]*(num_games/freq)
    draw_rate = [0]*(num_games/freq)

    alpha = 0.1 # learning rate
    for i in xrange(num_games):
        #print 'game #',i
        # play against epsilon-optimal player 
        S,R = play_opponent(v,index,opt_strat,eps_opt,eps_greedy)
        winner = R[-1]
        beta = rl.td_episode(S, R, phi, beta, alpha=alpha) # lam=0.9, gamma=1, alpha = 0.001
        v = numpy.dot(phi,beta)

        if (i % freq == (freq-1)):
            alpha = alpha/2
            eps_greedy = eps_greedy/2
            print 'alpha: ', alpha
            print 'eps greedy: ',eps_greedy
        
        if winner == 1:
            win_rate[i/freq]+=1
        elif winner == -1:
            lose_rate[i/freq]+=1
        else:
            draw_rate[i/freq]+=1
    
    for g in xrange(num_games/freq):
        win_rate[g] = win_rate[g]/float(freq)
        lose_rate[g] = lose_rate[g]/float(freq)
        draw_rate[g] = draw_rate[g]/float(freq)
    
    return beta, v, win_rate, lose_rate, draw_rate 


plac.annotations(
    out_path = ("path to write policy pickle",),
    states_path = ("path to TTT states pickle",),
    )

def make_optimal_policy(out_path = "ttt_optimal.pickle.gz", states_path = "ttt_states.pickle.gz"):
    """
    Generate the optimal TTT policy assuming that player 1 plays first. 
    Gives optimal move for player who is next to go given board state.
    """

    with contextlib.closing(gzip.GzipFile(states_path)) as pickle_file:
        states = pickle.load(pickle_file)

    policy = {}

    for (i, state) in enumerate(states):

        player = -2*numpy.sum(state._grid)+1
        move = ttt_optimal_move(state,player)
        policy[state] = move
        
        print 'player: ',player
        print "optimal move in state \n {0}: {1}".format(state._grid, move)

    with contextlib.closing(gzip.GzipFile(out_path, "w")) as pickle_file:
        pickle.dump(policy, pickle_file)


plac.annotations(
    k = ("number of features", "option", "k", int),
    num_games = ("number of games", "option", "n", int),
    freq = ("frequency of performance averaging","option","freq",int),
    eps_opt = ("fraction of moves played randomly by opponent rl agent trains \
        against", "option",float)
    )
def main(k=200,num_games=5000,freq=500, eps_opt=1, eps_greedy=0.3):

    beta, v, win_rate, lose_rate, draw_rate = train_rl_ttt_agent(k,num_games,freq,eps_opt,eps_greedy)
         
    with contextlib.closing(gzip.GzipFile("ttt_states.pickle.gz")) as pickle_file:
        states = pickle.load(pickle_file)
    index = dict(zip(states, xrange(len(states)))) 

    test_value_function(v,index,states) 
     
    # save the learned weight values
    with open("ttt_beta_laplacian_k="+str(k)+".pickle", "w") as pickle_file:
        pickle.dump(beta, pickle_file)

    x = range(freq,num_games+freq,freq)
    plt.plot(x,win_rate,'g-',x,lose_rate,'r-',x,draw_rate,'b-')
    plt.legend(('win rate', 'lose rate', 'draw rate'),loc=7)
    plt.xlabel('number of games')
    plt.ylabel('win,lose,draw rate - averaged every '+str(freq)+' games')
    if eps_opt == 1:
        plt.title('RL Agent Playing Against Random Tic Tac Toe Opponent')
    elif eps_opt == 0:
        plt.title('RL Agent Playing Against Optimal Tic Tac Toe Opponent')
    else:
        plt.title('RL Agent Playing Against Semi-Random (eps='+str(eps_opt)+') Tic Tac Toe Opponent')
    plt.show()



