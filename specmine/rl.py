import plac
from random import choice
import numpy
import specmine

class StateValueFunctionPolicy:
    def __init__(self,domain,values,epsilon=0):
        self._domain = domain
        self._values = values # split into weights and features?
        self._epsilon = epsilon

    def __getitem__(self,state):
        if numpy.random.random() < self._epsilon:
            moves = list(self._domain.actions_in(state))
            return choice(moves)
        else:
            max_value = None
            for action in self._domain.actions_in(state):
                after_state = self._domain.outcome_of(state,action)
                after_state_index = self._domain._index[after_state]
                value = self._values[after_state_index]

                if value > max_value:
                    best_moves = [action]
                    max_value = value
                elif value == max_value:
                    best_moves.append(action) 
        
            # choose randomly among the moves of highest value
            return choice(best_moves)

class RandomPolicy:
    def __init__(self,domain):
        self._domain = domain

    def __getitem__(self,state):
        print self._domain.actions_in(state)
        moves = list(self._domain.actions_in(state))
        return choice(moves)

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

def td_episode(S, R, phi, beta = None, lam=0.9, gamma=1, alpha = 0.001):

    k = phi.shape[1]
    if beta == None:
        beta = np.zeros(k)
    z = phi[S[0],:]   
    for t in xrange(len(R)):
        curr_phi = phi[S[t],:] 
        z_old = z
        if t == len(R)-1:
            # terminal state is defined as having value zero
            delta = z*(R[t]-np.dot(curr_phi,beta)) 

        else:
            delta = z*(R[t]+np.dot((gamma*phi[S[t+1],:]-curr_phi),beta))
            z = gamma*lam*z+phi[S[t+1],:]

        beta += delta*alpha/np.dot(z_old,curr_phi)

    return beta

def generate_episode(domain,policy):
    S = []; R = []
    state = domain.initial_state
    S.append(state)
    R.append(domain.reward_in(state))

    while not domain.check_end(state):
        action = policy[state]
        state = domain.outcome_of(state,action)
        S.append(state)
        R.append(domain.reward_in(state))
    
    return S,R

def main(num_episodes=10,k=10):
    
    print 'creating domain and opponent'
    rand_policy = RandomPolicy(None)
    ttt = specmine.domains.TicTacToeDomain(rand_policy)
    rand_policy._domain = ttt
    
    print 'generating representation'
    adj = specmine.tictac.adjacency_matrix()
    phi = specmine.spectral.laplacian_basis(adj,k, sparse=True)
    beta = numpy.zeros(k)
    print phi.shape
    print beta.shape
    v = numpy.dot(phi,beta)
    lvf_policy = StateValueFunctionPolicy(ttt,v)

    print 'learning new policy'
    S = []; R = []
    for i in xrange(num_episodes):
        s,r = generate_episode(ttt,lvf_policy)
        S.extend(s)
        R.extend(r)
    
    beta = td_episode(S, R, phi, beta = beta) # lam=0.9, gamma=1, alpha = 0.001
    v = numpy.dot(beta,phi)
    lvf_policy._values = v

if __name__ == '__main__':
    main()
