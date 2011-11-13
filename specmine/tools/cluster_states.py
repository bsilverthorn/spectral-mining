import plac
import specmine.tools.cluster_states

if __name__ == "__main__":
    plac.call(specmine.tools.cluster_states.main)

import cPickle as pickle
import specmine

@plac.annotations(
    out_path = ("path to write clustering",),
    states_path = ("path to states pickle",),
    clusters = ("number of clusters", "positional", None, int),
    )
def main(out_path, states_path, clusters):
    """Cluster a state space graph."""

    with specmine.util.openz(states_path) as pickle_file:
        boards = pickle.load(pickle_file)

    (indices, adjacency) = specmine.discovery.state_dict_to_adjacency(boards)
    clustering_array = specmine.graclus.cluster(adjacency, clusters)
    clustering_dict = dict((s, clustering_array[indices[s]]) for s in boards)

    with open(out_path, "wb") as out_file:
        pickle.dump(clustering_dict, out_file)

