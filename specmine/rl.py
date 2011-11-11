import plac
from random import choice
import numpy
import specmine


class StateValueFunctionPolicy:
    def __init__(self,domain,values,epsilon):
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
                after_state_index = self._domain.index[after_state]
                value = self._values[after_state_index]

                if value > max_value:]
                    best_moves = [action]
                    max_value = value
                elif value == max_value:)
                    best_moves.append(action) 
        
            # choose randomly among the moves of highest value
            return choice(moves)
