import csv
import cPickle as pickle
import numpy 
import matplotlib.pyplot as plt
import specmine

def main():

    k_vals = numpy.array(range(1,11))*50

    print 'creating domain and opponent'

    opponent_domain = specmine.rl.TicTacToeDomain(player = -1)
    opponent_policy = specmine.rl.RandomPolicy(opponent_domain)

    ttt = specmine.rl.TicTacToeDomain(player = 1, opponent = opponent_policy)
    

    pickle_path = specmine.util.static_path("ttt_states.pickle.gz")
    with specmine.util.openz(pickle_path) as pickle_file:
            adj_dict = pickle.load(pickle_file)
    adj_matrix, index = specmine.discovery.adjacency_dict_to_matrix(adj_dict)    
    num_states = len(index)

    w = csv.writer(file(specmine.util.static_path( \
        'feature_number_test.csv'),'wb'))
    w.writerow(['method','features','reward_mean','reward_variance'])
    plt.hold(True)

    laplacian_reward = [0]*len(k_vals)
    laplacian_variance = [0]*len(k_vals)
    for i in range(len(k_vals)):
        k = k_vals[i]
        print 'k: ', k
        laplacian_basis = specmine.spectral.laplacian_basis(adj_matrix,k, sparse=True)        
        laplacian_feature_map = specmine.discovery.TabularFeatureMap(laplacian_basis, index)    
        laplacian_reward[i], laplacian_variance[i] = specmine.science.evaluate_feature_map(laplacian_feature_map)

        random_basis = numpy.hstack((numpy.ones((num_states,1)),numpy.random.standard_normal((num_states,k-1))))        
        random_feature_map = specmine.discovery.TabularFeatureMap(random_basis, index) 
        random_reward[i], random_variance[i] = specmine.science.evaluate_feature_map(random_feature_map)
    
        w.writerow(['laplacian',k,laplacian_reward[i],laplacian_variance[i]])
        w.writerow(['random',k,random_reward[i],random_variance[i]])

    plt.plot(k_vals,laplacian_reward,k_vals,random_reward)
    plt.legend(('laplacian','random'))
    plt.show()


if __name__ == '__main__':
    main()
