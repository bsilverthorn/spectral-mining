import cPickle as pickle
import numpy
import specmine


def main(num_episodes=10,k=10):
    
    print 'creating domain and opponent'
    rand_policy = specmine.rl.RandomPolicy(None)
    ttt = specmine.domains.TicTacToeDomain(rand_policy)
    rand_policy.domain = ttt

    pickle_path = specmine.util.static_path("ttt_states.pickle.gz")
    with specmine.util.openz(pickle_path) as pickle_file:
            adj_dict = pickle.load(pickle_file)

    print 'generating representation'
    adj_matrix, index = specmine.discovery.adjacency_dict_to_matrix(adj_dict)
    print index[(specmine.tictac.BoardState(),1)]
    phi = specmine.spectral.laplacian_basis(adj_matrix,k, sparse=True)
    feature_map = specmine.discovery.TabularFeatureMap(phi,index)

    weights = numpy.zeros(k)
    value_function = specmine.rl.LinearValueFunction(feature_map,weights)
    lvf_policy = specmine.rl.StateValueFunctionPolicy(ttt,value_function)

    print 'learning new policy'
    S = []; R = []
    for i in xrange(num_episodes):
        s,r = specmine.rl.generate_episode(ttt,lvf_policy)
        S.extend(s)
        R.extend(r)
    
    value_function.weights = specmine.rl.td_episode(S, R, phi, beta = beta) # lam=0.9, gamma=1, alpha = 0.001

if __name__ == '__main__':
    main()

