import plac
import specmine

if __name__ == '__main__':
    plac.call(specmine.tools.write_ttt_states.main)

import json
import cPickle as pickle
import numpy
import tictac

@plac.annotations(
    out_path = ("path to write states pickle",),
    start = ("start state", "option", None, json.loads),
    cutoff = ("move limit", "option", None, int),
    )
def main(out_path, start = None, cutoff = None):
    """Generate and write the TTT board adjacency map."""

    if start is None:
        start_board = tictac.BoardState()
    else:
        start_board = tictac.BoardState(numpy.array(start))

    states = tictac.construct_adjacency_dict(start_board, cutoff = cutoff)

    with specmine.util.openz(out_path, "wb") as pickle_file:
        pickle.dump(states, pickle_file)

