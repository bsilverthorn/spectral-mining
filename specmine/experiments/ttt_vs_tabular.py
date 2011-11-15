import csv
import cPickle as pickle
import numpy
import matplotlib.pyplot as plt
import specmine


def evaluate_policy(domain, policy, episodes):
    rewards = []
    for i in xrange(episodes):
        s,r = specmine.rl.generate_episode(domain,policy)
        rewards.append(r[-1])

    return numpy.mean(rewards), numpy.var(rewards)

def get_learning_curve(domain,basis,index,num_evals,games_per_eval, games_between,alpha):

    feature_map = specmine.discovery.TabularFeatureMap(basis, index)
    reward = [0]*num_evals
    variance = [0]*num_evals
    policy = specmine.rl.StateValueFunctionPolicy(domain,specmine.rl.LinearValueFunction(feature_map)) 
    reward[0],variance[0] = evaluate_policy(domain,policy,games_per_eval)
    for i in xrange(1,num_evals):
        policy = specmine.rl.linear_td_learn_policy(domain, feature_map, episodes = games_between, weights = policy.values.weights, alpha = alpha)
        reward[i],variance[i] = evaluate_policy(domain,policy,games_per_eval)

    return reward, variance

def main(K=[50,100,200,300], num_evals = 250, games_per_eval = 500, games_between = 2, alpha = 0.001):
    
    print 'creating domain and opponent'

    opponent_domain = specmine.rl.TicTacToeDomain(player = -1)
    opponent_policy = specmine.rl.RandomPolicy(opponent_domain)

    ttt = specmine.rl.TicTacToeDomain(player = 1, opponent = opponent_policy)
    
    print 'generating representation'

    pickle_path = specmine.util.static_path("ttt_states.pickle.gz")
    with specmine.util.openz(pickle_path) as pickle_file:
            adj_dict = pickle.load(pickle_file)

    adj_matrix, index = specmine.discovery.adjacency_dict_to_matrix(adj_dict)
    num_states = len(index)

    x = numpy.array(range(num_evals))*games_between
    w = csv.writer(file(specmine.util.static_path( \
        'learning_curve.'+str(num_evals*games_between)+'games.alpha='+str(alpha)+'.gpe='+str(games_per_eval)+'.csv'),'wb'))
    w.writerow(['method','features','games','reward_mean','reward_variance'])
    plt.hold(True)

    print 'evaluating tabular representation'
    # run and record tabular
    tab = numpy.eye(num_states)
    reward_tabular,var_tabular  = get_learning_curve(ttt, tab, index, num_evals, games_per_eval, games_between,alpha)
    for i in xrange(len(reward_tabular)):
        w.writerow(['tabular', None, x[i], reward_tabular[i], var_tabular[i]])
    #plt.plot(x,reward_tabular)
    #plt.errorbar(x,reward_tabular,yerr=var_tabular)

    print 'evaluating linear features'
    # laplacian and random for different values of k
    for k in K:
        print 'laplacian for ', k, ' features'
        phi = specmine.spectral.laplacian_basis(adj_matrix,k, sparse=True)
        reward_laplacian,var_laplacian = get_learning_curve(ttt, phi, index, num_evals, games_per_eval, games_between,alpha)
        for i in xrange(len(reward_laplacian)):
            w.writerow(['laplacian',k,x[i],reward_laplacian[i], var_laplacian[i]])
        #plt.plot(x,reward_laplacian)
        #plt.errorbar(x,reward_laplacian,yerr=var_laplacian)

        print 'random for ', k, ' features'
        rand = numpy.hstack((numpy.ones((num_states,1)),numpy.random.standard_normal((num_states,k-1))))
        reward_random,var_random  = get_learning_curve(ttt, rand, index, num_evals, games_per_eval, games_between,alpha)
        for i in xrange(len(reward_random)):
            w.writerow(['random',k,x[i],reward_random[i], var_random[i]]) 
        #plt.plot(x,reward_random)
        #plt.errorbar(x,reward_random,yerr=var_random)

    #plt.legend(['tabular','laplacian k='+str(K[0]),'random k='+str(K[0]),'laplacian k='+str(K[1]),'random k='+str(K[1]),'laplacian k='+str(K[2]),'random k='+str(K[2])])
    #plt.show()
    
if __name__ == '__main__':
    main()

