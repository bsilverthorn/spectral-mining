import specmine.experiments.cluster_ttt

if __name__ == "__main__":
    specmine.script(specmine.experiments.cluster_ttt.main)

import csv
import numpy
import scipy.sparse
import sklearn.cluster
import sklearn.neighbors
<<<<<<< HEAD
import condor
=======
import specmine
>>>>>>> 34b74182af48b7ae75f0c44aba20829717c2188d

logger = specmine.get_logger(__name__)

def raw_state_features((board, player)):
    return [1] + list(board.grid.flatten())

def xxx_feature_map(adict):
    pass

def affinity_graph(vectors_ND, neighbors):
    """Build the k-NN affinity graph from state feature vectors."""

    G = neighbors
    (N, D) = vectors_ND.shape

    # find nearest neighbors
    logger.info("finding nearest neighbors in affinity space")

    tree = sklearn.neighbors.BallTree(vectors_ND)
    (neighbor_distances_NG, neighbor_indices_NG) = tree.query(vectors_ND, k = G)

    # construct the affinity graph
    logger.info("constructing the affinity graph")

    affinity_lil_NN = scipy.sparse.lil_matrix((N, N))

    for n in xrange(N):
        for g in xrange(G):
            m = neighbor_indices_NG[n, g]

            affinity_lil_NN[n, m] = affinity_lil_NN[m, n] = neighbor_distances_NG[n, g]

    return affinity_lil_NN

    ## cluster states
    #logger.info("aliasing states with spectral clustering")

    #clustering = sklearn.cluster.SpectralClustering(mode = "arpack")

    #clustering.fit(affinity_lil_NN.tocsr())

    ## cluster states directly
    #K = clusters
    #clustering = sklearn.cluster.KMeans(k = K)

    #clustering.fit(features_ND)

def evaluate_vs_b(B, vectors_ND, affinity_NN):
    if B > 0:
        affinity_basis_NB = specmine.spectral.laplacian_basis(affinity_NN, B)
        all_features_NF = numpy.hstack([vectors_ND, affinity_basis_NB])
    else:
        all_features_NF = vectors_ND

    feature_map = specmine.discovery.TabularFeatureMap(all_features_NF, index)
    (mean, variance) = specmine.science.evaluate_feature_map(feature_map)

    return [B, mean, variance]

@specmine.annotations(
    out_path = ("path to write CSV",),
    clusters = ("number of clusters", "option", None, int),
    neighbors = ("number of neighbors", "option", None, int),
    )
def main(out_path, clusters = 16, neighbors = 8):
    """Run TTT state-clustering experiment(s)."""

    # convert states to their vector representations
    states = list(specmine.tictac.load_adjacency_dict())

    logger.info("converting states to their vector representation")

    vectors_ND = numpy.array(map(raw_state_features, states))

    def yield_jobs():
        for B in numpy.r_[0:200:16j].astype(int):
            yield (evaluate_vs_b, [B, vectors_ND, affinity_NN])

    with open(out_path, "wb") as out_file:
        out_csv = csv.writer(out_file)

        # build the affinity graph
        affinity_NN = affinity_graph(vectors_ND, neighbors)

    with open(out_path, "w") as out_file:
        writer = csv.writer(out_file)

        writer.writerow(["basis_vectors", "mean_reward", "reward_variance"])

        cargo.do_or_distribute(yield_jobs(), workers, lambda _, r: writer.writerow(r))

