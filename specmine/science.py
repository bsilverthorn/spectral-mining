import numpy
import specmine

def evaluate_feature_map(feature_map):
    opponent_domain = specmine.rl.TicTacToeDomain(player = -1)
    opponent_policy = specmine.rl.RandomPolicy(opponent_domain)

    domain = specmine.rl.TicTacToeDomain(player = 1, opponent = opponent_policy)

    games_for_learning = 10000
    games_for_testing = 500

    policy = \
        specmine.rl.linear_td_learn_policy(
            domain,
            feature_map,
            episodes = games_for_learning,
            )

    rewards = []

    for i in xrange(games_for_testing):
        (s, r) = specmine.rl.generate_episode(domain,policy)

        rewards.append(r[-1])

    return (numpy.mean(rewards), numpy.var(rewards))

