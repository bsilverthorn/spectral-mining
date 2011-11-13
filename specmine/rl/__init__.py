from .td import *
from .domains import *
from .policies import *
from .q_learning import *
from .v_learning import *

def generate_episode(domain,policy):
    S = []; R = []
    state = domain.initial_state
    S.append(state)
    R.append(domain.reward_in(state))

    while not domain.check_end(state):
        action = policy[state]
        state = domain.outcome_of(state,action) # assumes deterministic

        S.append(state)
        R.append(domain.reward_in(state))
    
    return S, R

