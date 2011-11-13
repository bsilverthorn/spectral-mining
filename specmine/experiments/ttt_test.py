import cPickle as pickle
import numpy
import matplotlib.pyplot as plt
import specmine


def evaluate_policy(domain, policy, episodes):
    rewards = []
    for i in xrange(episodes):
        S,R = specmine.rl.generate_episode(domain,policy)
        rewards.append(R[-1])

    return numpy.mean(rewards)


def main(num_episodes=10, k=100, num_evals = 100, games_per_eval = 100, games_between = 50):
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
    reward = [0]*num_evals
    for i in xrange(num_evals):
        print i 
        policy = specmine.rl.linear_td_learn_policy(ttt, feature_map, episodes = games_between)
        reward[i] = evaluate_policy(ttt,policy,games_per_eval)
        print reward[i]

    
    
    plt.plot(reward)
    plt.show()
    
if __name__ == '__main__':
    main()

