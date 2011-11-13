import plac
import specmine.tools.draw_states

if __name__ == "__main__":
    plac.call(specmine.tools.draw_states.main)

import cPickle as pickle
import colorsys
import tempfile
import subprocess
import numpy
import specmine

def write_dot_file(out_file, states, directed = False, coloring = None):
    # prepare
    names = dict((s, "s{0}".format(i)) for (i, s) in enumerate(states))

    if coloring is None:
        coloring = dict((s, 0) for s in states)

    unique = len(set(coloring.values()))
    hues = numpy.r_[0.0:1.0 - 1.0 / unique:unique * 1j]

    # write general configuration
    if directed:
        out_file.write("digraph G {\n")

        edge = "->"
    else:
        out_file.write("graph G {\n")

        edge = "--"

    out_file.write("node [label=\"\",shape=point,color=\"#00002299\"];\n")
    out_file.write("edge [color=\"#00000022\"];\n")
    out_file.write("splines=true;\n")

    #import tictac
    #out_file.write("root={0};\n".format(names[tictac.BoardState()]))

    # write the nodes
    for (i, state) in enumerate(states):
        values = colorsys.hsv_to_rgb(hues[coloring[state]], 0.85, 0.85)
        string = "#{0}bb".format("".join("{0:02x}".format(int(round(v * 255.0))) for v in values))

        out_file.write("{0} [color=\"{1}\"];\n".format(names[state], string))

    # write the edges
    for (state, next_states) in states.iteritems():
        for next_state in next_states:
            out_file.write("{0} {2} {1};\n".format(names[state], names[next_state], edge))

    # ...
    out_file.write("}")

def render_dot_file(out_path, dot_path, tool_name):
    command = [
        tool_name,
        "-v",
        "-Tpdf",
        "-o",
        out_path,
        dot_path
        ]

    print "running:", command

    subprocess.check_call(command)

@plac.annotations(
    out_path = ("path to write graphviz dot file",),
    states_path = ("path to state space pickle",),
    render_with = ("graphviz rendering tool", "option", "r"),
    coloring_path = ("path to vertex-color map", "option", "c"),
    )
def main(out_path, states_path, render_with = None, coloring_path = None):
    """Visualize a state space graph."""

    with specmine.util.openz(states_path) as pickle_file:
        boards = pickle.load(pickle_file)

    print "writing {1}-vertex graph to {0}".format(out_path, len(boards))

    if coloring_path is None:
        coloring = None
    else:
        with specmine.util.openz(coloring_path) as pickle_file:
            coloring = pickle.load(pickle_file)

        assert len(coloring) == len(boards)

    if render_with is None:
        with open(out_path, "wb") as out_file:
            write_dot_file(out_file, boards, coloring = coloring)
    else:
        with tempfile.NamedTemporaryFile(suffix = ".dot") as dot_file:
            write_dot_file(dot_file, boards, coloring = coloring)

            dot_file.flush()

            render_dot_file(out_path, dot_file.name, render_with)

