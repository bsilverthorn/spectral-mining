import plac
import specmine

if __name__ == "__main__":
    plac.call(specmine.tools.stateviz.main)

import cPickle as pickle

@plac.annotations(
    out_path = ("path to write graphviz dot file",),
    states_path = ("path to state space pickle",),
    )
def main(out_path, states_path):
    """Visualize a state space graph."""

    with specmine.util.openz("ttt_states.pickle.gz") as pickle_file:
        boards = pickle.load(pickle_file)

    print type(boards)

    print boards[0]

