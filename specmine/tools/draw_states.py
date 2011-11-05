import plac
import specmine

if __name__ == "__main__":
    plac.call(specmine.tools.stateviz.main)

import cPickle as pickle
import tictac

def write_dot_file(out_file, states, directed = False):
    # map states to node names
    names = dict((s, "s{0}".format(i)) for (i, s) in enumerate(states))

    # write the header, and prepare
    if directed:
        out_file.write("digraph G {\n")

        edge = "->"
    else:
        out_file.write("graph G {\n")

        edge = "--"

    # write the nodes
    out_file.write("center={0};\n".format(names[tictac.BoardState()]))

    for (i, state) in enumerate(states):
        out_file.write("{0} [label=\"\",shape=point];\n".format(names[state]))

    # write the edges
    for (state, next_states) in states.iteritems():
        for next_state in next_states:
            out_file.write("{0} {2} {1};\n".format(names[state], names[next_state], edge))

    # ...
    out_file.write("}")

@plac.annotations(
    out_path = ("path to write graphviz dot file",),
    states_path = ("path to state space pickle",),
    )
def main(out_path, states_path):
    """Visualize a state space graph."""

    with specmine.util.openz("ttt_states.pickle.gz") as pickle_file:
        boards = pickle.load(pickle_file)

    with open(out_path, "wb") as out_file:
        write_dot_file(out_file, boards)

