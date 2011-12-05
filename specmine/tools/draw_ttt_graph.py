import cPickle as pickle
import tempfile
import specmine

logger = specmine.get_logger(__name__)

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

if __name__ == "__main__":
    specmine.script(main)

