import colorsys
import tempfile
import subprocess
import numpy
import specmine

logger = specmine.get_logger(__name__)

def write_dot_file(out_file, states, directed = False, coloring = None, root = None):
    # prepare
    names = dict((s, "s{0}".format(i)) for (i, s) in enumerate(states))

    # write general configuration
    if directed:
        out_file.write("digraph G {\n")

        edge = "->"
    else:
        out_file.write("graph G {\n")

        edge = "--"

    #out_file.write("node [label=\"\",width=0.15,height=0.15];\n")
    out_file.write("node [label=\"\"];\n")
    out_file.write("edge [color=\"#00000022\"];\n")
    out_file.write("splines=true;\n")
    out_file.write("outputorder=edgesfirst;\n")

    if root is not None:
        #out_file.write("root={0};\n".format(names[(specmine.tictac.BoardState(), 1)]))
        out_file.write("root={0};\n".format(names[root]))

    # write the nodes
    for (i, state) in enumerate(states):
        #(board, player) = state
        #attributes = []

        #if player == 1:
            #attributes.append("shape=point,color=\"#00ff00\"")
        #else:
            #attributes.append("shape=square,color=\"#0000ff\",style=filled")

        attributes = ["shape=point"]

        if coloring is not None:
            attributes.append("color=\"{0}\"".format(coloring[state]))

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

    logger.info("running: %s", command)

    subprocess.check_call(command)

def continuous_to_coloring(values, hue_min = 0.0, hue_max = 0.9):
    values = numpy.array(values)
    #values -= numpy.min(values[values > numpy.min(values)])
    #values /= numpy.max(values[values < numpy.max(values)])
    values -= numpy.min(values)
    values /= numpy.max(values)

    print values

    def value_to_color(value):
        rgbs = colorsys.hsv_to_rgb(value, 0.85, 0.85)
        #if value > 1.0:
            #value = 1.0
        #elif value < 0.0:
            #value = 0.0
        #rgbs = colorsys.hsv_to_rgb(0.0, 0.0, value)
        string = "#{0}".format("".join("{0:02x}".format(int(round(v * 255.0))) for v in rgbs))

        return string

    return map(value_to_color, values)

def categorical_to_coloring(XXX):
    # XXX
    if coloring is None:
        coloring = {}
    else:
        unique = len(set(coloring.values()))
        hues = numpy.r_[0.0:1.0 - 1.0 / unique:unique * 1j]

    if h is not None:
        values = colorsys.hsv_to_rgb(hues[h], 0.85, 0.85)
        string = "#{0}bb".format("".join("{0:02x}".format(int(round(v * 255.0))) for v in values))

        attributes.append("color=" + string)

def visualize_graph(out_path, states, render_with = None, coloring = None, root = None):
    if render_with is None:
        with open(out_path, "wb") as out_file:
            write_dot_file(out_file, states, coloring = coloring, root = root)
    else:
        with tempfile.NamedTemporaryFile(suffix = ".dot") as dot_file:
            write_dot_file(dot_file, states, coloring = coloring, root = root)

            dot_file.flush()

            render_dot_file(out_path, dot_file.name, render_with)

