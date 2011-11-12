import scipy.sparse

class TabularFeatureMap:
    def __init__(self, basis_matrix, index):
        self.basis = basis_matrix # number of states x number of features
        self.index = index

    def __getitem__(self,state):
        return self.basis[index[state],:]

def adjacency_dict_to_matrix(adict):
    """Create a symmetric adjacency matrix from a state -> [state] dict."""

    N = len(adict)
    index = dict(zip(adict, xrange(N)))
    amatrix = scipy.sparse.lil_matrix((N, N), dtype = int)

    for node in adict:
        n = index[node]

        for neighbor in adict[node]:
            m = index[neighbor]

            amatrix[n, m] = 1
            amatrix[m, n] = 1

    return (amatrix.tocsr(), index)

