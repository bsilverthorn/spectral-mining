import numpy

class SimpleRoomDomain(object):
    def __init__(self, width = 8):
        self._width = width
        self._actions = map(numpy.array, [[1, 0], [-1, 0], [0, 1], [0, -1], [0, 0]])

    def actions_in(self, (state_x, state_y)):
        for (action_x, action_y) in self._actions:
            if 0 <= state_x + action_x < self._width and 0 <= state_y + action_y < self._width:
                yield (action_x, action_y)

    def reward_in(self, (x, y)):
        if x == 0 and y == 0:
            return 0
        else:
            return -1

    def outcome_of(self, (state_x, state_y), (action_x, action_y)):
        return (state_x + action_x, state_y + action_y)

    @property
    def states(self):
        for x in xrange(self._width):
            for y in xrange(self._width):
                yield (x, y)

class QFunctionPolicy(object):
    def __init__(self, domain, q_values):
        self._domain = domain
        self._q_values = q_values

def learn_q_policy(domain, rate = 1e-1, discount = 9e-1, iterations = 1000):
    q_values = {}

    for state in domain.states:
        for action in domain.actions_in(state):
            q_values[(state, action)] = 0.0

    for _ in xrange(iterations):
        for state in domain.states:
            for action in domain.actions_in(state):
                # compute the action value
                next_state = domain.outcome_of(state, action)
                max_next_value = max(q_values[(next_state, a)] for a in domain.actions_in(next_state))

                # update our table
                value = q_values[(state, action)]
                error = domain.reward_in(next_state) + discount * max_next_value - value

                q_values[(state, action)] = value + rate * error

    return QFunctionPolicy(domain, q_values)

def main():
    # build the state space
    domain = SimpleRoomDomain()
    policy = learn_q_policy(domain)

if __name__ == "__main__":
    main()

