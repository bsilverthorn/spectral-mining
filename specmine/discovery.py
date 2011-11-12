import scipy.sparse

def adjacency_dict_to_matrix(adict):
    """Create a symmetric adjacency matrix from a state -> [state] dict."""

    N = len(adict)
    index = dict(zip(adict, xrange(N)))
    amatrix = scipy.sparse.lil_matrix((N, N), dtype = int)

    for node in adict:
        n = index[node]

        for neighbor in adict[node]:
            m = index[neighbor]

            amatrix[m, n] = 1
            amatrix[n, n] = 1

    return amatrix.to_csr()

