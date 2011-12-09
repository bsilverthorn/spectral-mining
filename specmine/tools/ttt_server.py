import json
import cPickle as pickle
import numpy
import sklearn
import flask
import specmine

logger = specmine.get_logger(__name__)
server = flask.Flask(__name__)

server.debug = True

global_policy = None
global_feature_map = None
global_value_function = None

@server.route("/analyze_board")
def analyze_board():
    # disassemble request
    player = int(flask.request.args["player"])
    grid = numpy.array(json.loads(flask.request.args["board"])).reshape((3, 3)).astype(float)
    board = specmine.tictac.BoardState(grid)

    # apply the value functions
    numbers = numpy.argsort(numpy.abs(global_value_function.weights))[-6:][::-1]
    #numbers = numpy.arange(6, dtype = int)
    weights = global_value_function.weights[numbers]
    values = numpy.empty((1 + numbers.shape[0], 9), object)

    for i in xrange(3):
        for j in xrange(3):
            if grid[i, j] == 0:
                next_state = (board.make_move(player, i, j), -1 * player)

                h = i * 3 + j
                n = global_feature_map.index[next_state]

                values[0, h] = global_value_function[next_state]

                for b in xrange(numbers.shape[0]):
                    values[b + 1, h] = global_feature_map.basis[n, numbers[b]]

                if player == -1:
                    values[:, h] *= -1

    # ...
    return \
        flask.jsonify(
            values = values.tolist(),
            numbers = numbers.tolist(),
            weights = weights.tolist(),
            )

def raw_state_features((board, player)):
    return [1] + list(board.grid.flatten())

def prepare_state(values_path):
    B = 200

    with specmine.openz(values_path) as values_file:
        values = pickle.load(values_file)

    logger.info("converting states to their vector representation")

    states_adict = specmine.tictac.load_adjacency_dict()
    (gameplay_NN, gameplay_index) = specmine.discovery.adjacency_dict_to_matrix(states_adict)
    basis_NB = specmine.spectral.laplacian_basis(gameplay_NN, B)
    feature_map = specmine.discovery.TabularFeatureMap(basis_NB, gameplay_index)

    # construct domain
    opponent_domain = specmine.rl.TicTacToeDomain(player = -1)
    opponent_policy = specmine.rl.RandomPolicy(opponent_domain)
    domain = specmine.rl.TicTacToeDomain(player = 1, opponent = opponent_policy)

    # prepare features and targets
    states = list(values)

    state_features = numpy.array([feature_map[s] for s in states])
    state_values = numpy.array([values[s] for s in states])

    # learn a value function
    logger.info("fitting value function predictor")

    ridge = sklearn.linear_model.Ridge(alpha = 1.0)

    ridge.fit(state_features, state_values)

    value_function = specmine.rl.LinearValueFunction(feature_map, ridge.coef_)
    policy = specmine.rl.StateValueFunctionPolicy(domain, value_function)

    return (policy, feature_map, value_function)

def main(values_path = None):
    """Serve TTT with a learned policy."""

    global global_policy
    global global_feature_map
    global global_value_function

    state_path = "server_state.pickle.gz"

    if values_path is None:
        logger.info("loading precomputed server state")

        with specmine.openz(state_path) as values_file:
            state = pickle.load(values_file)

        logger.info("done loading server state")
    else:
        state = prepare_state(values_path)

        with specmine.openz(state_path, "wb") as state_file:
            pickle.dump(state, state_file)

    (global_policy, global_feature_map, global_value_function) = state

    server.run(host = "0.0.0.0")

if __name__ == "__main__":
    specmine.script(main)

