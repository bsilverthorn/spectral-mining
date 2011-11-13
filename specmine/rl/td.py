import numpy
import specmine

def td_episode(S, R, features, beta = None, lam=0.9, gamma=1, alpha = 0.001):
    z = features[S[0]]

    if beta == None:
        beta = numpy.zeros_like(z)

    for t in xrange(len(R)):
        curr_phi = features[S[t]]
        z_old = z

        if t == len(R)-1:
            # terminal state is defined as having value zero
            delta = z*(R[t] - numpy.dot(curr_phi,beta)) 
        else:
            delta = z*(R[t]+numpy.dot((gamma*features[S[t+1]]-curr_phi),beta))
            z = gamma * lam * z + features[S[t + 1]]

        beta += delta*alpha/numpy.dot(z_old,curr_phi)

    return beta

def lstd_episode(S, R, phi, lam=0.9, A=None, b=None):

    k = phi.shape[1] # number of features
    if A == None:
        A = np.zeros((k,k))
    if b == None:
        b = np.zeros((k,1))
    z = phi[S[0],:] # eligibility trace initialized to first state features
    for t in xrange(len(R)):
        A += np.dot(z[:,None],(phi[S[t],:]-phi[S[t+1],:])[None,:])
        b += R[t]*z[:,None]
        z = lam*z+phi[S[t+1],:]

    return A, b

def lstd_solve(A,b):

    beta = np.linalg.solve(A,b) # solve for feature parameters
    return beta

def linear_td_learn_policy(domain, features, episodes = 1, **kwargs):
    """Learn a linear TD policy."""

    weights = None

    rewards = []

    for i in xrange(episodes):
        value_function = specmine.rl.LinearValueFunction(features, weights)
        lvf_policy = specmine.rl.StateValueFunctionPolicy(domain, value_function)
        S, R = specmine.rl.generate_episode(domain, lvf_policy)
        weights = specmine.rl.td_episode(S, R, features, beta = weights, **kwargs)

        rewards.append(R[-1])

        if i % 200 == 0:
            print numpy.mean(rewards)
            rewards = []

    return lvf_policy

