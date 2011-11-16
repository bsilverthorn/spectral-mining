import specmine.experiments.ttt_feature_number

if __name__ == '__main__':
    specmine.script(specmine.experiments.ttt_feature_number.main)

import csv
import cPickle as pickle
import numpy 
import condor
import specmine

def run_laplacian_evaluation(k, adj_matrix, index):
    laplacian_basis = specmine.spectral.laplacian_basis(adj_matrix,k, sparse=True)        
    laplacian_feature_map = specmine.discovery.TabularFeatureMap(laplacian_basis, index)    
    reward, variance = specmine.science.evaluate_feature_map_lstd(laplacian_feature_map)

    return ["laplacian", k, reward, variance]

def run_random_evaluation(k, adj_matrix, index):
    num_states = len(index)
    random_basis = numpy.hstack((numpy.ones((num_states,1)),numpy.random.standard_normal((num_states,k-1))))        
    random_feature_map = specmine.discovery.TabularFeatureMap(random_basis, index) 
    reward, variance = specmine.science.evaluate_feature_map_lstd(random_feature_map)

    return ["random", k, reward, variance]

def main(workers=0):

    print 'creating domain and opponent'

    pickle_path = specmine.util.static_path("ttt_states.pickle.gz")
    with specmine.util.openz(pickle_path) as pickle_file:
            adj_dict = pickle.load(pickle_file)
    adj_matrix, index = specmine.discovery.adjacency_dict_to_matrix(adj_dict)    

    w = csv.writer(file(specmine.util.static_path( \
        'feature_number_test_lstd.csv'),'wb'))
    w.writerow(['method','features','reward_mean','reward_variance'])

    def yield_jobs():
        for k in numpy.array(range(1,11))*50:
            yield (run_laplacian_evaluation, [k, adj_matrix, index])
            yield (run_random_evaluation, [k, adj_matrix, index])

    condor.do_or_distribute(yield_jobs(), workers, lambda _, r: w.writerow(r))

    #plt.hold(True)
    #plt.plot(k_vals,laplacian_reward,k_vals,random_reward)
    #plt.legend(('laplacian','random'))
    #plt.show()

