import nose
import specmine

example_directed = {
    "a": ["b", "c"],
    "b": ["c"],
    "c": ["a"],
    }
example_undirected = {
    "a": ["b", "c"],
    "b": ["a", "c"],
    "c": ["a", "b"],
    }
example_once = {
    "a": ["b", "c"],
    "b": [],
    "c": ["b"],
    }

def test_adjacency_dict_to_matrix():
    # build the matrix
    (amatrix, index) = specmine.discovery.adjacency_dict_to_matrix(example_directed)

    # hand-construct the expected matrix
    alist = [[0] * 3 for _ in xrange(3)]

    ai = index["a"]
    bi = index["b"]
    ci = index["c"]

    alist[ai][bi] = alist[bi][ai] = 1
    alist[ai][ci] = alist[ci][ai] = 2
    alist[bi][ci] = alist[ci][bi] = 1

    # compare
    nose.tools.assert_equal(amatrix.todense().tolist(), alist)

def test_adjacency_matrix_to_dict_directed():
    (amatrix, index) = specmine.discovery.adjacency_dict_to_matrix(example_directed)
    rindex = dict((v, k) for (k, v) in index.iteritems())
    adict = specmine.discovery.adjacency_matrix_to_dict(amatrix, rindex, make_directed = True)

    for key in adict:
        adict[key] = sorted(adict[key])

    nose.tools.assert_equal(adict, example_once)

def test_adjacency_matrix_to_dict_undirected():
    (amatrix, index) = specmine.discovery.adjacency_dict_to_matrix(example_directed)
    rindex = dict((v, k) for (k, v) in index.iteritems())
    adict = specmine.discovery.adjacency_matrix_to_dict(amatrix, rindex, make_directed = False)

    for key in adict:
        adict[key] = sorted(adict[key])

    nose.tools.assert_equal(adict, example_undirected)

