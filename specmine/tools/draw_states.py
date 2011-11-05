import plac
import specmine

if __name__ == "__main__":
    plac.call(specmine.tools.draw_states.main)

import cPickle as pickle
import colorsys
import tempfile
import subprocess
import numpy

def write_dot_file(out_file, states, directed = False, coloring = None):
    # prepare
    names = dict((s, "s{0}".format(i)) for (i, s) in enumerate(states))

    if coloring is None:
        coloring = dict((s, 0) for s in states)

    hues = numpy.r_[0.0:1.0:len(set(coloring.values())) * 1j]

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

    # write the nodes
    for (i, state) in enumerate(states):
        (rgb_r, rgb_g, rgb_g) = colorsys.hsv_to_rgb(hues[coloring[state]], 0.75, 1.0)

        out_file.write("{0} [color=\"{1} 1.0 1.0\"];\n".format(names[state], hue))

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
    coloring_path = ("path to vertex-color map", "option", "c"),
    render_with = ("graphviz rendering tool", "option", "r"),
    )
def main(out_path, states_path, coloring_path = None, render_with = None):
    """Visualize a state space graph."""

    with specmine.util.openz("ttt_states.pickle.gz") as pickle_file:
        boards = pickle.load(pickle_file)

    print "writing {1}-vertex graph to {0}".format(out_path, len(boards))

    if coloring_path is None:
        coloring = None
    else:
        with specmine.util.openz(coloring_path) as pickle_file:
            coloring = pickle.load(pickle_file)

    if render_with is None:
        with open(out_path, "wb") as out_file:
            write_dot_file(out_file, boards, coloring = coloring)
    else:
        with tempfile.NamedTemporaryFile(suffix = ".dot") as dot_file:
            write_dot_file(dot_file, boards)

            dot_file.flush()

            render_dot_file(out_path, dot_file.name, render_with)

