from random import choice

class QFunctionPolicy(object):
    def __init__(self, domain, q_values):
        self._domain = domain
        self._q_values = q_values

    def __getitem__(self, state):
        return \
            max(
                self._domain.actions_in(state),
                key = lambda a: self._q_values[(state, a)],
                )

class StateValueFunctionPolicy:
    def __init__(self,domain,value_function,epsilon=0):
        self.domain = domain
        self.values = values_function # split into weights and features?
        self.epsilon = epsilon

    def __getitem__(self,state):
        if numpy.random.random() < self.epsilon:
            moves = list(self._domain.actions_in(state))
            return choice(moves)
        else:
            max_value = None
            for action in self.domain.actions_in(state):
                after_state = self.domain.outcome_of(state,action)
                after_state_index = self.domain.index[after_state]
                value = self.values[after_state_index]

                if value > max_value:
                    best_moves = [action]
                    max_value = value
                elif value == max_value:
                    best_moves.append(action) 
        
            # choose randomly among the moves of highest value
            return choice(best_moves)

class RandomPolicy:
    def __init__(self,domain):
        self.domain = domain

    def __getitem__(self,state):
        print self.domain.actions_in(state)
        moves = list(self.domain.actions_in(state))
        return choice(moves)

