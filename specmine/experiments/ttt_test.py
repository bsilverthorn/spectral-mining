import cPickle as pickle
import numpy
import specmine


def main(num_episodes=10,k=100):
    print 'creating domain and opponent'

    opponent_domain = specmine.rl.TicTacToeDomain(player = -1)
    opponent_policy = specmine.rl.RandomPolicy(opponent_domain)

    ttt = specmine.rl.TicTacToeDomain(player = 1, opponent = opponent_policy)

    print 'generating representation'

    pickle_path = specmine.util.static_path("ttt_states.pickle.gz")
    with specmine.util.openz(pickle_path) as pickle_file:
            adj_dict = pickle.load(pickle_file)

    adj_matrix, index = specmine.discovery.adjacency_dict_to_matrix(adj_dict)
    phi = specmine.spectral.laplacian_basis(adj_matrix,k, sparse=True)
    feature_map = specmine.discovery.TabularFeatureMap(phi, index)

    print 'learning new policy'

    policy = specmine.rl.linear_td_learn_policy(ttt, feature_map, episodes = 10000)

if __name__ == '__main__':
    main()

