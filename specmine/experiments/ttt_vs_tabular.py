import csv
import cPickle as pickle
import numpy
import specmine

def evaluate_policy(domain, policy, episodes):
    rewards = []
    for i in xrange(episodes):
        s,r = specmine.rl.generate_episode(domain,policy)
        rewards.append(r[-1])

    return numpy.mean(rewards), numpy.var(rewards)

def get_td_learning_curve(domain,features,num_evals,games_per_eval, games_between,alpha):

    reward = [0]*num_evals
    variance = [0]*num_evals
    policy = specmine.rl.StateValueFunctionPolicy(domain,specmine.rl.LinearValueFunction(features)) 
    reward[0],variance[0] = evaluate_policy(domain,policy,games_per_eval)
    for i in xrange(1,num_evals):
        policy = specmine.rl.linear_td_learn_policy(domain, features, episodes = games_between, weights = policy.values.weights, alpha = alpha)
        reward[i],variance[i] = evaluate_policy(domain,policy,games_per_eval)

    return reward, variance

def get_lstd_learning_curve(domain, features,games_per_iter, num_iters, games_per_eval):

    reward = [0]*num_iters
    variance = [0]*num_iters
    policy = specmine.rl.StateValueFunctionPolicy(domain,specmine.rl.LinearValueFunction(features)) 
    reward[0],variance[0] = evaluate_policy(domain,policy,games_per_eval)
    weights = None
    A = None; b = None
    for i in xrange(1,num_iters):
        policy, A, b = specmine.rl.lstd_learn_policy(domain, features, games_per_iter, 1, weights=weights, epsilon = 0.1, A=A, b=b,reg=0.2,decay=1-2e-5)
        
        weights = policy.values.weights
        
        reward[i],variance[i] = evaluate_policy(domain, policy, games_per_eval)
        print 'reward: ', reward[i]

    return reward, variance

#def main(K=[100], num_evals = 250, games_per_eval = 500, games_between = 2, alpha = 0.001):
def main(K=[100], games_per_iter=500, num_iters = 20, games_per_eval = 500):

    print 'creating domain and opponent'

    opponent_domain = specmine.rl.TicTacToeDomain(player = -1)
    opponent_policy = specmine.rl.RandomPolicy(opponent_domain)

    ttt = specmine.rl.TicTacToeDomain(player = 1, opponent = opponent_policy)
    
    print 'generating representation'

    pickle_path = specmine.util.static_path("ttt_states.pickle.gz")
    with specmine.util.openz(pickle_path) as pickle_file:
            adj_dict = pickle.load(pickle_file)
    adj_matrix, index = specmine.discovery.adjacency_dict_to_matrix(adj_dict)

    w = csv.writer(file(specmine.util.static_path( \
        #'learning_curve.'+str(num_evals*games_between)+'games.alpha='+str(alpha)+'.gpe='+str(games_per_eval)+'.csv'),'wb'))
        'learning_curve_lstd.csv'),'wb'))
    w.writerow(['method','features','games','reward_mean','reward_variance'])

#    print 'evaluating tabular representation'
#    # run and record tabular
#    tab = numpy.eye(num_states)
#    reward_tabular,var_tabular  = get_lstd_learning_curve(ttt, tab, index, num_evals, games_per_eval, games_between,alpha)
#    for i in xrange(len(reward_tabular)):
#        w.writerow(['tabular', None, x[i], reward_tabular[i], var_tabular[i]])

    print 'evaluating linear features'
    # laplacian and random for different values of k
    for k in K:
        print 'laplacian for ', k, ' features'
        phi = specmine.spectral.laplacian_basis(adj_matrix,k, sparse=True)
        features = specmine.discovery.TabularFeatureMap(phi,index)
        reward_laplacian,var_laplacian = get_lstd_learning_curve(ttt, features, games_per_iter, num_iters, games_per_eval)
        for i in xrange(len(reward_laplacian)):
            w.writerow(['laplacian',k,i,reward_laplacian[i], var_laplacian[i]])

#        print 'random for ', k, ' features'
#        rand = numpy.hstack((numpy.ones((num_states,1)),numpy.random.standard_normal((num_states,k-1))))
#        reward_random,var_random  = get_learning_curve(ttt, rand, index, num_evals, games_per_eval, games_between,alpha)
#        for i in xrange(len(reward_random)):
#            w.writerow(['random',k,x[i],reward_random[i], var_random[i]]) 


    #plt.legend(['tabular','laplacian k='+str(K[0]),'random k='+str(K[0]),'laplacian k='+str(K[1]),'random k='+str(K[1]),'laplacian k='+str(K[2]),'random k='+str(K[2])])
    #plt.show()
    
if __name__ == '__main__':
    main()
