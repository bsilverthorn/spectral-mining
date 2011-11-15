import numpy
import specmine

logger = specmine.get_logger(__name__)

def td_episode(S, R, features, beta = None, lam = 0.9, gamma = 1.0, alpha = 1e-3):
    next_phi = features[S[0]]
    z = next_phi

    if beta == None:
        beta = numpy.zeros_like(next_phi)

    deltas = []
    first_beta = numpy.copy(beta)

    for t in xrange(len(R)):
        curr_phi = next_phi

        if t == len(R)-1:
            # terminal state is defined as having value zero
            delta = R[t] - numpy.dot(curr_phi, beta)
        else:
            next_phi = features[S[t + 1]]

            delta = R[t] + numpy.dot((gamma * next_phi - curr_phi), beta)

        z *= gamma * lam
        z += curr_phi

        beta += delta * alpha * z

        deltas.append(delta)

        #inner = numpy.dot(z_old, curr_phi)

        #if inner > 0.0:
            #beta += delta * alpha / inner

    print ",".join(map(str, [
        alpha,
        numpy.sum(numpy.abs(beta - first_beta)),
        numpy.mean(deltas),
        R[-1],
        ]))

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

def linear_td_learn_policy(domain, features, episodes = 1, weights = None, **kwargs):
    """Learn a linear TD policy starting with the given weights."""

    print "alpha,change,mean_error,reward"

    #alpha = 1e-1
    alpha = 1e-2
    epsilon = 1.0

    for i in xrange(episodes):
        if i % 1000 == 0:
            logger.info("learning linear TD policy; iteration %i", i)

        value_function = specmine.rl.LinearValueFunction(features, weights)
        lvf_policy = specmine.rl.StateValueFunctionPolicy(domain, value_function, epsilon = epsilon)
        S, R = specmine.rl.generate_episode(domain, lvf_policy)
        #weights = specmine.rl.td_episode(S, R, features, beta = weights, **kwargs)
        weights = specmine.rl.td_episode(S, R, features, beta = weights, alpha = alpha, **kwargs)

        #alpha *= 1.0 - 1e-4
        epsilon *= 1.0 - 1e-3

    value_function = specmine.rl.LinearValueFunction(features, weights)
    lvf_policy = specmine.rl.StateValueFunctionPolicy(domain, value_function)

    return lvf_policy

