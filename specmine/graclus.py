import os.path
import shutil
import tempfile
import subprocess
import contextlib
import numpy
import scipy.sparse

@contextlib.contextmanager
def mkdtemp_scoped(prefix = ""):
    """Create, and then delete, a temporary directory."""

    path = None

    try:
        path = tempfile.mkdtemp(prefix = prefix)

        yield path
    finally:
        if path is not None:
            shutil.rmtree(path, ignore_errors = True)

def write_graph(out_file, adjacency):
    """Write an undirected graph to disk in graclus format."""

    # build a sparse matrix without self edges
    sparse_adjacency = scipy.sparse.lil_matrix(adjacency)

    if sparse_adjacency.dtype.kind not in "iu":
        raise Exception("edges must have integer weights for clustering")

    (M, N) = sparse_adjacency.shape

    sparse_adjacency.setdiag(numpy.zeros(N))

    E = sparse_adjacency.nnz / 2

    assert M == N
    assert sparse_adjacency.nnz == 2 * E

    # write the graph file
    out_file.write("{0} {1} 1\n".format(N, E))

    for n in xrange(N):
        line = zip(sparse_adjacency.rows[n], sparse_adjacency.data[n])

        out_file.write(" ".join(str(m + 1) + " " + str(e) for (m, e) in line) + "\n")

def cluster(adjacency, clusters, extra_options = ()):
    """Cluster an undirected graph with graclus."""

    with mkdtemp_scoped("graclus.") as working_root:
        graph_path = os.path.join(working_root, "graph")

        with open(graph_path, "w") as graph_file:
            write_graph(graph_file, adjacency)

            graph_file.flush()

            command = [
                os.path.abspath(os.path.join(os.path.dirname(__file__), "../graclus1.2/graclus")),
                os.path.abspath(graph_file.name),
                str(clusters),
                ]

            output_path = os.path.join(working_root, "graclus.out")

            with open(output_path, "wb") as output_file:
                subprocess.check_call(
                    command,
                    cwd = working_root,
                    stdout = output_file,
                    stderr = subprocess.STDOUT,
                    )

            clustering_path = "{0}.part.{1}".format(graph_path, clusters)

            if os.path.exists(clustering_path):
                with open(clustering_path) as clustering_file:
                    return numpy.array(map(int, clustering_file.read().split()))
            else:
                with open(output_path) as output_file:
                    raise Exception("clustering failed; output follows.\n{0}".format(output_file.read()))

def test_cluster():
    adjacency = [
        [1, 0, 1],
        [0, 1, 0],
        [1, 0, 0],
        ]
    clustering = cluster(adjacency, 3)

    assert len(clustering) == 3
    assert clustering[0] != clustering[1]
    assert clustering[0] != clustering[2]
    assert clustering[1] != clustering[2]

