import specmine
import specmine.experiments.cluster_ttt

if __name__ == "__main__":
    specmine.script(specmine.experiments.cluster_ttt.main)

import numpy
import scipy.sparse
import sklearn.cluster
import sklearn.neighbors

logger = specmine.get_logger(__name__)

def raw_state_features((board, player)):
    return board.grid.flatten()

def evaluate_feature_map(feature_map):
    opponent_domain = specmine.rl.TicTacToeDomain(player = -1)
    opponent_policy = specmine.rl.RandomPolicy(opponent_domain)

    domain = specmine.rl.TicTacToeDomain(player = 1, opponent = opponent_policy)

    games_for_learning = 10000
    games_for_testing = 500

    policy = \
        specmine.rl.linear_td_learn_policy(
            domain,
            feature_map,
            episodes = games_for_learning,
            )

    rewards = []

    for i in xrange(games_for_testing):
        (s, r) = specmine.rl.generate_episode(domain,policy)

        rewards.append(r[-1])

    return (numpy.mean(rewards), numpy.var(rewards))

#def xxx_feature_map(adict):
    ## find nearest neighbors
    #logger.info("finding nearest neighbors")

    #G = neighbors
    #tree = sklearn.neighbors.BallTree(features_ND)
    #(neighbor_distances_NG, neighbor_indices_NG) = tree.query(features_ND, k = G)

    ## construct the adjacency matrix
    #logger.info("constructing the adjacency matrix")

    #affinity_lil_NN = scipy.sparse.lil_matrix((N, N))

    #for state in states:
        #n = index[state]

        #for g in xrange(G):
            #m = neighbor_indices_NG[n, g]

            #affinity_lil_NN[n, m] = affinity_lil_NN[m, n] = neighbor_distances_NG[n, g]

    ## cluster states
    #logger.info("aliasing states with spectral clustering")

    #clustering = sklearn.cluster.SpectralClustering(mode = "arpack")

    #clustering.fit(affinity_lil_NN.tocsr())

@specmine.annotations(
    clusters = ("number of clusters", "option", None, int),
    neighbors = ("number of neighbors", "option", None, int),
    )
def main(clusters = 16, neighbors = 8):
    """Cluster TTT states."""

    ## convert states to their vector representations
    #states = list(specmine.tictac.load_adjacency_dict())

    #logger.info("converting states to their vector representation")

    #index = dict(zip(states, xrange(len(states))))
    #features_ND = numpy.array(map(raw_state_features, states))
    #(N, D) = features_ND.shape

    ## cluster states directly
    #K = clusters
    #clustering = sklearn.cluster.KMeans(k = K)

    #clustering.fit(features_ND)

