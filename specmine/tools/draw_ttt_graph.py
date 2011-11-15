import specmine.tools.draw_ttt_graph

if __name__ == "__main__":
    specmine.script(specmine.tools.draw_ttt_graph.main)

import cPickle as pickle
import colorsys
import tempfile
import subprocess
import numpy
import specmine

logger = specmine.get_logger(__name__)

def write_dot_file(out_file, states, directed = False, coloring = None):
    # prepare
    names = dict((s, "s{0}".format(i)) for (i, s) in enumerate(states))

    if coloring is None:
        coloring = {}
    else:
        unique = len(set(coloring.values()))
        hues = numpy.r_[0.0:1.0 - 1.0 / unique:unique * 1j]

    # write general configuration
    if directed:
        out_file.write("digraph G {\n")

        edge = "->"
    else:
        out_file.write("graph G {\n")

        edge = "--"

    out_file.write("node [label=\"\",width=0.15,height=0.15];\n")
    out_file.write("edge [color=\"#00000022\"];\n")
    out_file.write("splines=true;\n")
    out_file.write("outputorder=edgesfirst;\n")
    out_file.write("root={0};\n".format(names[(specmine.tictac.BoardState(), 1)]))

    # write the nodes
    for (i, state) in enumerate(states):
        (board, player) = state
        attributes = []

        if player == 1:
            attributes.append("shape=point,color=\"#00ff00\"")
        else:
            attributes.append("shape=square,color=\"#0000ff\",style=filled")

        h = coloring.get(state)

        if h is not None:
            values = colorsys.hsv_to_rgb(hues[h], 0.85, 0.85)
            string = "#{0}bb".format("".join("{0:02x}".format(int(round(v * 255.0))) for v in values))

            attributes.append("color=" + string)

        out_file.write("{0} [{1}];\n".format(names[state], ",".join(attributes)))

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

@specmine.annotations(
    out_path = ("path to write graphviz dot file",),
    states_path = ("path to adjacency dict", "option",),
    render_with = ("graphviz rendering tool", "option", "r"),
    coloring_path = ("path to vertex-color map", "option", "c"),
    )
def main(out_path, states_path = None, render_with = None, coloring_path = None):
    """Visualize a state space graph."""

    if states_path is None:
        states = specmine.tictac.load_adjacency_dict()
    else:
        with specmine.openz(states_path) as states_file:
            states = pickle.load(states_file)

    logger.info("writing %i-vertex graph to %s", len(states), out_path)

    if coloring_path is None:
        coloring = None
    else:
        with specmine.util.openz(coloring_path) as pickle_file:
            coloring = pickle.load(pickle_file)

        assert len(coloring) == len(states)

    if render_with is None:
        with open(out_path, "wb") as out_file:
            write_dot_file(out_file, states, coloring = coloring)
    else:
        with tempfile.NamedTemporaryFile(suffix = ".dot") as dot_file:
            write_dot_file(dot_file, states, coloring = coloring)

            dot_file.flush()

            render_dot_file(out_path, dot_file.name, render_with)

