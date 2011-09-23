import csv
import cPickle as pickle
import numpy
import scipy.sparse
import scipy.sparse.linalg
import spectral

class BoardState(object):
    
    def __init__(self, grid):
        self._string = str(grid)
        self._grid = grid

    def __hash__(self):
        return hash(self._string)

    def __eq__(self,other):
        return self._string == other._string

    def make_move(self,player,i,j):
        new_grid = self._grid.copy()
        new_grid[i,j] = player

        return BoardState(new_grid)

    def check_end(self):
        return \
            numpy.any(numpy.sum(self._grid, axis = 0) == 3) \
            or numpy.any(numpy.sum(self._grid, axis = 1) == 3) \
            or numpy.sum(numpy.diag(self._grid)) == 3 \
            or (self._grid[0,2] + self._grid[1,1] + self._grid[2,0]) == 3

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
                    board_recurs(-1*player, board, board.make_move(player, i,j))
    
    board_recurs(1, None, init_board)
    
    return states
    
def main():
    #states = construct_adjacency()

    #with open("states.pickle", "w") as pickle_file:
        #pickle.dump(states, pickle_file)

    with open("states.pickle") as pickle_file:
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

        #init_board = BoardState(numpy.zeros((3,3)))

        #csv_writer.writerow(["i","j","eigen_index","value"])

        #for i in xrange(3):
            #for j in xrange(3):
                #next_board = init_board.make_move(1, i, j)

                #for k in xrange(9):
                    #csv_writer.writerow([i, j, k, v[index[next_board], k]])

        csv_writer.writerow(["index", "i","j","piece"])

        largest = v[:, 3].argsort()
        for k in xrange(16):
            board = rindex[largest[-(k + 1)]]

            for i in xrange(3):
                for j in xrange(3):
                    csv_writer.writerow([k, i, j, board._grid[i, j]])

        #csv_writer.writerow(["eigen_index", "board", "value"])

        #largest = v[:, 1].argsort()
        #for n in xrange(N):
            #board = rindex[largest[n]]

            #for e in xrange(4):
                #csv_writer.writerow([e, n, v[index[board], e]])

if __name__ == '__main__':
    main()

