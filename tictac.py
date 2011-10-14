import plac
import tictac

if __name__ == '__main__':
    plac.call(tictac.main_write_states_pickle)

import csv
import gzip
import cPickle as pickle
import contextlib
import numpy
import scipy.sparse
import scipy.sparse.linalg
import spectral

class BoardState(object):
    # XXX use a simple bitstring board representation?
    # XXX an efficient end-state check could be made against a mask set

    def __init__(self, grid):
        self._string = str(grid)
        self._grid = grid

    def __hash__(self):
        return hash(self._string)

    def __eq__(self, other):
        return self._string == other._string

    def make_move(self, player, i, j):
        assert self._grid[i, j] == 0

        new_grid = self._grid.copy()

        new_grid[i,j] = player

        return BoardState(new_grid)

    def get_winner(self):
        """Return the winner of the board, if any."""

        # implemented explicitly for easy Cythonization

        # draw?
        board_product = 1

        for i in xrange(3):
            board_product *= self._grid[i, 0] * self._grid[i, 1] * self._grid[i, 2]

        if board_product != 0:
            return 0

        # row or column win?
        for i in xrange(3):
            row_sum = self._grid[i, 0] + self._grid[i, 1] + self._grid[i, 2]
            col_sum = self._grid[0, i] + self._grid[1, i] + self._grid[2, i]

            if row_sum == 3 or col_sum == 3:
                return 1
            elif row_sum == -3 or col_sum == -3:
                return -1

        # diagonal win?
        tb_diag_sum = self._grid[0, 0] + self._grid[1, 1] + self._grid[2, 2]
        bt_diag_sum = self._grid[0, 2] + self._grid[1, 1] + self._grid[2, 0]

        if tb_diag_sum == 3 or bt_diag_sum == 3:
            return 1
        elif tb_diag_sum == -3 or bt_diag_sum == -3:
            return -1

        # no win, game still in progress
        return None

    def check_end(self):
        """Is this board state an end state?"""

        return self.get_winner() != None

def test_board_state():
    configurations = [
        [[0, 0, 0], [0, 0, 0], [0, 0, 0]],
        [[1, 0, 0], [0, 1, 0], [0, 0, 1]],
        [[0, -1, 0], [0, -1, 0], [0, -1, 0]],
        [[0, 0, 1], [-1, -1, -1], [0, 1, 0]],
        [[1, 1, -1], [-1, 1, -1], [1, -1, 1]],
        ]
    boards = [BoardState(numpy.array(c)) for c in configurations]

    assert boards[0].get_winner() is None
    assert boards[0].make_move(1, 2, 2)._grid[2, 2] == 1
    assert boards[0]._grid[2, 2] == 0
    assert boards[0] == boards[0]
    assert boards[0] != boards[1]
    assert boards[1].get_winner() == 1
    assert boards[2].get_winner() == -1
    assert boards[3].get_winner() == -1
    assert boards[4].get_winner() == 0

def construct_adjacency():
    
    states = {}
    init_board = BoardState(numpy.zeros((3,3)))
    
    def board_recurs(player, parent, board):
        print len(states)
        adjacent = states.get(board)
        
        if adjacent is None:
            adjacent = states[board] = set()
        
        if parent is not None:
            adjacent.add(parent)
            
        if board.check_end():
            return
        
        for i in xrange(3):
            for j in xrange(3):
                if board._grid[i,j] == 0:
                    board_recurs(-1 * player, board, board.make_move(player, i, j))
    
    board_recurs(1, None, init_board)
    
    return states

def get_ttt_laplacian_basis(k=100): 
    ''' calculate the first k eigenvectors of the ttt graph laplacian and 
    return them as columns of the matrix phi along with the dictionary of 
    state indices'''

    #states = construct_adjacency()

    with contextlib.closing(gzip.GzipFile("states.pickle.gz")) as pickle_file:
        states = pickle.load(pickle_file)
    
    index = dict(zip(states, xrange(len(states))))
    N = len(index)

    adjacency_NN = numpy.zeros((N, N))

    for state in index:
        n = index[state]

        for parent in states[state]:
            adjacency_NN[index[parent], n] = 1
            adjacency_NN[n, index[parent]] = 1

    laplacian_NN = spectral.laplacian_operator(adjacency_NN)

    print "finding eigenvalues"

    splaplacian_NN = scipy.sparse.csr_matrix(laplacian_NN)
    (lam, phi) = scipy.sparse.linalg.eigen_symmetric(splaplacian_NN, k, which = "SM")
    sort_inds = lam.argsort()
    lam = lam[sort_inds]
    phi = phi[:, sort_inds]
    print 'is resulting basis sparse?: ',scipy.sparse.issparse(phi)
    
    return phi, index

@plac.annotations(
    out_path = ("path to write states pickle (gzipped)",)
    )
def main_write_states_pickle(out_path):
    """Generate and write the TTT board adjacency map."""

    states = construct_adjacency()

    with contextlib.closing(gzip.GzipFile(out_path, "w")) as pickle_file:
        pickle.dump(states, pickle_file)

def main_visualize():
    with gzip.GzipFile("states.pickle.gz") as pickle_file:
        states = pickle.load(pickle_file)

    index = dict(zip(states, xrange(len(states))))
    rindex = sorted(states, key = lambda s: index[s])
    N = len(index)

    adjacency_NN = numpy.zeros((N, N))

    for state in index:
        n = index[state]

        for parent in states[state]:
            adjacency_NN[index[parent], n] = 1
            adjacency_NN[n, index[parent]] = 1

    laplacian_NN = spectral.laplacian_operator(adjacency_NN)

    print "finding eigenvalues"

    splaplacian_NN = scipy.sparse.csr_matrix(laplacian_NN)

    (lam, v) = scipy.sparse.linalg.eigen_symmetric(splaplacian_NN, k = 8, which = "SM")
    sort_inds = lam.argsort()
    lam = lam[sort_inds]
    v = v[:, sort_inds]

    print "visualizing eigenvectors"

    with open("spectrum.csv", "w") as csv_file:
        csv_writer = csv.writer(csv_file)

        ## first visualization
        #init_board = BoardState(numpy.zeros((3,3)))

        #csv_writer.writerow(["i","j","eigen_index","value"])

        #for i in xrange(3):
            #for j in xrange(3):
                #next_board = init_board.make_move(1, i, j)

                #for k in xrange(9):
                    #csv_writer.writerow([i, j, k, v[index[next_board], k]])

        # second visualization
        csv_writer.writerow(["index", "i","j","piece"])

        largest = v[:, 3].argsort()
        for k in xrange(16):
            board = rindex[largest[-(k + 1)]]

            for i in xrange(3):
                for j in xrange(3):
                    csv_writer.writerow([k, i, j, board._grid[i, j]])

        ## third visualization
        #csv_writer.writerow(["eigen_index", "board", "value"])

        #largest = v[:, 1].argsort()
        #for n in xrange(N):
            #board = rindex[largest[n]]

            #for e in xrange(4):
                #csv_writer.writerow([e, n, v[index[board], e]])

