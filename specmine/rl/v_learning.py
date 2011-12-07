import specmine

logger = specmine.get_logger(__name__)

def compute_state_values_table(domain, discount = 1.0, iterations = 128):
    """Run tabular value iteration on a finite domain."""

    logger.info(
        "running value iteration for %i iterations on %i states",
        iterations,
        len(domain.states),
        )

    values = {}

    for state in domain.states:
        values[state] = 0.0

    for i in xrange(iterations):
        total_change = 0.0

        for state in domain.states:
            # compute the value of acting in this state
            max_action_value = 0.0

            for action in domain.actions_in(state):
                action_value = discount * values[domain.outcome_of(state, action)]

                if action_value > max_action_value:
                    max_action_value = action_value

            # update our table
            new_value = domain.reward_in(state) + max_action_value

            total_change += abs(values[state] - new_value)

            values[state] = new_value

        print total_change

        if total_change < 1e-16:
            break

    return values

def compute_state_values_table_nondet(domain, discount = 1.0, iterations = 128):
    """Run tabular value iteration on a finite domain."""

    # XXX TTT-specific, and hard-coded for the random opponent

    logger.info(
        "running value iteration for %i iterations on %i states",
        iterations,
        len(domain.states),
        )

    values = {}

    for state in domain.states:
        values[state] = 0.0

    for i in xrange(iterations):
        total_change = 0.0

        for state in domain.states:
            # compute the value of acting in this state
            max_action_value = 0.0

            for action in domain.actions_in(state):
                action_value = 0.0

                for (next_state, p) in domain.outcomes_of(state, action):
                    action_value += p * values[next_state]

                if action_value > max_action_value:
                    max_action_value = action_value

            # update our table
            new_value = domain.reward_in(state) + discount * max_action_value

            total_change += abs(values[state] - new_value)

            values[state] = new_value

        logger.info("total change in value iteration: %f", total_change)

        if total_change < 1e-16:
            break

    return values

