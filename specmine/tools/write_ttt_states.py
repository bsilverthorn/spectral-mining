import specmine.tools.write_ttt_states

if __name__ == '__main__':
    specmine.script(specmine.tools.write_ttt_states.main)

import json
import cPickle as pickle
import numpy
import specmine

def sampled_adjacency_dict(num_games = 100, sample_policy = None, opponent_policy=None):
    
    if opponent_policy == None:
        opponent_domain = specmine.rl.TicTacToeDomain(player = -1)
        opponent_policy = specmine.rl.RandomPolicy(opponent_domain)

    domain = specmine.rl.TicTacToeDomain(player = 1, opponent = opponent_policy) 
    if sample_policy == None:
        sample_policy = specmine.rl.RandomPolicy(domain)
    
    adict = {}
    for i in xrange(num_games):
        S,_ = specmine.rl.generate_episode(domain,sample_policy)
        for i in xrange(len(S)):
            adjacent = adict.get(S[i])
            
            if adjacent is None:
                adjacent = adict[S[i]] = set()

            if i < (len(S)-1):
                adjacent.add(S[i+1])
    return adict

@specmine.annotations(
    out_path = ("path to write states pickle",),
    start = ("start state", "option", None, json.loads),
    cutoff = ("move limit", "option", None, int),
    sampled = ("use sampled paths", "flag"),
    )
def main(out_path, start = None, cutoff = None, sampled = False):
    """Generate and write the TTT board adjacency map."""

    if start is None:
        start_board = specmine.tictac.BoardState()
    else:
        start_board = specmine.tictac.BoardState(numpy.array(start))

    if sampled:
        states = sampled_adjacency_dict()
    else:
        states = specmine.tictac.construct_adjacency_dict(start_board, cutoff = cutoff)

    with specmine.util.openz(out_path, "wb") as pickle_file:
        pickle.dump(states, pickle_file)

