import numpy
import specmine

logger = specmine.get_logger(__name__)

def evaluate_feature_map(feature_map):
    # construct domain
    opponent_domain = specmine.rl.TicTacToeDomain(player = -1)
    opponent_policy = specmine.rl.RandomPolicy(opponent_domain)

    domain = specmine.rl.TicTacToeDomain(player = 1, opponent = opponent_policy)

    # learn a policy
    games_for_learning = 10

    logger.info("learning TTT policy over %i games", games_for_learning)

    policy = \
        specmine.rl.linear_td_learn_policy(
            domain,
            feature_map,
            episodes = games_for_learning,
            )

    # evaluate the policy
    games_for_testing = 5
    rewards = []

    logger.info("evaluating TTT policy over %i games", games_for_testing)

    for i in xrange(games_for_testing):
        (s, r) = specmine.rl.generate_episode(domain, policy)

        rewards.append(r[-1])

    return (numpy.mean(rewards), numpy.var(rewards))

