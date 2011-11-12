import gzip
import cPickle as pickle
import numpy
import play_tictac
import tictac
import rl

def test_td_backups(num_runs=10,k=50,eps_opt=1,eps_greedy=0):
    phi, index = tictac.get_ttt_laplacian_basis(k)
    n = phi.shape[0] # number of states
    beta = numpy.zeros(k) # numpy.random.standard_normal(k);
    v = numpy.dot(phi,beta)
    
    print 'testing td backup on one repeated episode'

    # load optimal strategy
    with gzip.GzipFile("ttt_optimal.pickle") as pickle_file:
        opt_strat = pickle.load(pickle_file)
    R = [0]
    while R[-1] == 0:
        S,R = play_tictac.play_opponent(v,index,opt_strat,eps_opt,eps_greedy)
    
    #print 'true value of final state: ', R[-1]
    alpha = 1 # learning rate
    for i in xrange(num_runs):
        beta = rl.td_episode(S, R, phi, beta, alpha=alpha) # lam=0.9, gamma=1
        v = numpy.dot(phi,beta)
        # print 'approximate value of final state: ', v[S[-1]]

    assert (v[S[-1]] - R[-1]) < 10**-4
        
    

def main():
    test_td_backups()


if __name__ == "__main__":
    main()
