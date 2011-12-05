import go

def main(num_games=10):
    
    boards = set() 
    for i in xrange(num_games):
        S,R = go.read_expert_episode(sgf_path)
        boards.update(S)

