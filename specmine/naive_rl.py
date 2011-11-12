import plac
import specmine

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

def learn_q_values(domain, rate = 1e-1, discount = 9e-1, iterations = 128):
    q_values = {}

    for state in domain.states:
        for action in domain.actions_in(state):
            q_values[(state, action)] = 0.0

    for i in xrange(iterations):
        for state in domain.states:
            for action in domain.actions_in(state):
                # compute the action value
                next_state = domain.outcome_of(state, action)
                max_next_value = None

                for next_action in domain.actions_in(next_state):
                    next_value = q_values[(next_state, next_action)]

                    if max_next_value is None or max_next_value < next_value:
                        max_next_value = next_value

                # update our table
                value = q_values[(state, action)]

                if max_next_value is None:
                    error = domain.reward_in(next_state) - value
                else:
                    error = domain.reward_in(next_state) + discount * max_next_value - value

                q_values[(state, action)] = value + rate * error

                print ",".join(map(str, [i, error]))

    return q_values

def learn_q_policy(domain, **options):
    q_values = learn_q_values(domain, **options)

    return QFunctionPolicy(domain, q_values)

def main():
    # build the state space
    domain = specmine.domains.TicTacToeDomain()
    policy = learn_q_policy(domain)

if __name__ == "__main__":
    plac.call(main)

