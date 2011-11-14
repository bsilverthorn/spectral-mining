import specmine.experiments.cluster_ttt

if __name__ == "__main__":
    specmine.script(specmine.experiments.cluster_ttt.main)

import csv
import numpy
import random
import scipy.sparse
import sklearn.cluster
import sklearn.neighbors
import condor
import specmine

logger = specmine.get_logger(__name__)

def raw_state_features((board, player)):
    return [1] + list(board.grid.flatten())

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

    coo_is = []
    coo_js = []
    coo_distances = []

    for n in xrange(N):
        for g in xrange(G):
            m = neighbor_indices_NG[n, g]

            coo_is.append(n)
            coo_js.append(m)
            coo_distances.append(neighbor_distances_NG[n, g])

    coo_distances = numpy.array(2 * coo_distances)
    coo_affinities = numpy.exp(-coo_distances**2 / 2.0)

    return scipy.sparse.coo_matrix((coo_affinities, (coo_is + coo_js, coo_js + coo_is)))

    ## cluster states
    #logger.info("aliasing states with spectral clustering")

    #clustering = sklearn.cluster.SpectralClustering(mode = "arpack")

    #clustering.fit(affinity_lil_NN.tocsr())

    ## cluster states directly
    #K = clusters
    #clustering = sklearn.cluster.KMeans(k = K)

    #clustering.fit(features_ND)

def evaluate_vs_b(B, vectors_ND, affinity_NN, index):
    if condor.get_task() is not None:
        numpy.random.seed(hash(condor.get_task().key))
        random.seed(2 * hash(condor.get_task().key))

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
    workers = ("number of condor jobs", "option", None, int),
    )
def main(out_path, clusters = 16, neighbors = 8, workers = 0):
    """Run TTT state-clustering experiment(s)."""

    # convert states to their vector representations
    states = list(specmine.tictac.load_adjacency_dict())

    logger.info("converting states to their vector representation")

    index = dict(zip(states, xrange(len(states))))
    #vectors_ND = numpy.array(map(raw_state_features, states))
    #affinity_NN = affinity_graph(vectors_ND, neighbors)

    indicator_features_NN = numpy.eye(len(index))
    feature_map = specmine.discovery.TabularFeatureMap(indicator_features_NN, index)
    (mean, variance) = specmine.science.evaluate_feature_map(feature_map)

    #def yield_jobs():
        #for B in numpy.r_[0:500:16j].astype(int):
        ##for B in numpy.r_[0:100:16j].astype(int):
        ##for B in [100]:
            #yield (evaluate_vs_b, [B, vectors_ND, affinity_NN, index])

    #with open(out_path, "wb") as out_file:
        #writer = csv.writer(out_file)

        #writer.writerow(["basis_vectors", "reward_mean", "reward_variance"])

        #condor.do_or_distribute(yield_jobs(), workers, lambda _, r: writer.writerow(r))

