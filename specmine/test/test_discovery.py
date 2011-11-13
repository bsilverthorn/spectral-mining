import nose
import specmine

def test_adjacency_dict_to_matrix():
    # build the matrix
    adict = {
        "a": ["b", "c"],
        "b": ["c"],
        "c": ["a"],
        }
    (amatrix, index) = specmine.discovery.adjacency_dict_to_matrix(adict)

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

