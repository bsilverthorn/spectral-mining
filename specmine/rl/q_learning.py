import specmine

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

    return specmine.rl.QFunctionPolicy(domain, q_values)

