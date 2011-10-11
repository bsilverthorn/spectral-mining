# naive RL implementation(s)

import random
import numpy

def main():
    # build the state space
    width = 8
    adjacency = numpy.zeros((width, width, width, width), numpy.uint8)

    for x in xrange(width):
        for y in xrange(width):
            if x > 0:
                adjacency[x, y, x - 1, y] = adjacency[x - 1, y, x, y] = 1
            if x < width - 1:
                adjacency[x, y, x + 1, y] = adjacency[x + 1, y, x, y] = 1
            if y > 0:
                adjacency[x, y, x, y - 1] = adjacency[x, y - 1, x, y] = 1
            if y < width - 1:
                adjacency[x, y, x, y + 1] = adjacency[x, y + 1, x, y] = 1

    actions = map(numpy.array, [[1, 0], [-1, 0], [0, 1], [0, -1], [0, 0]])

    def applicable_actions(state):
        for (a, action) in enumerate(actions):
            next_state = state + action

            if not numpy.any((next_state >= width) | (next_state < 0)):
                yield (a, action)

    # set up the rewards
    rewards = -1 * numpy.ones((width, width))

    rewards[0, 0] = 0

    # run Q-learning
    alpha = 1e-1
    lambda_ = 9e-1
    steps = 100000
    state = numpy.array([width - 1, width - 1])
    q_values = numpy.zeros((width, width, len(actions)))

    for i in xrange(steps):
        # taken a random action
        (x, y) = state
        (a, action) = random.choice(list(applicable_actions(state)))
        next_state = state + action
        state = next_state

        # compute the next-state max reward
        (next_x, next_y) = next_state
        max_next_value = -numpy.inf

        for (b, _) in applicable_actions(next_state):
            next_value = q_values[next_x, next_y, b]

            if max_next_value <= next_value:
                max_next_value = next_value

        # update our table
        q_values[x, y, a] = q_values[x, y, a] + alpha * (rewards[next_x, next_y] + lambda_ * max_next_value - q_values[x, y, a])

    numpy.set_printoptions(precision = 2)

    for j in xrange(len(actions)):
        print q_values[..., j]

if __name__ == "__main__":
    main()

