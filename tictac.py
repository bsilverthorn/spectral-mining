import numpy

class BoardState(object):
    
    def __init__(self, grid):
        self._hash = int(numpy.sum(grid))
        self._grid = grid
        
    def __hash__(self):
        
        return self._hash
    
    def __eq__(self,other):
        return (self._grid == other).all()
        
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
    states = construct_adjacency()
    print states
    
if __name__ == '__main__':
    main()

                    
                    