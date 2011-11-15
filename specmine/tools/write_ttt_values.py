import specmine.tools.write_ttt_values

if __name__ == "__main__":
    specmine.script(specmine.tools.write_ttt_values.main)

import cPickle as pickle
import specmine

logger = specmine.get_logger(__name__)

specmine.annotations(
    out_path = ("path to write values pickle",),
    opponent_path = ("path to opponent policy; random otherwise",)
    )
def main(out_path, opponent_path = None):
    """Compute the TTT value function using value iteration."""

    # construct domain
    opponent_domain = specmine.rl.TicTacToeDomain(player = -1)

    if opponent_path is None:
        opponent_policy = specmine.rl.RandomPolicy(opponent_domain)
    else:
        logger.info("loading opponent policy from %s", opponent_path)

        with specmine.openz(opponent_path) as opponent_file:
            opponent_policy = pickle.load(opponent_file)

    domain = specmine.rl.TicTacToeDomain(player = 1, opponent = opponent_policy)

    # compute the value function
    values = specmine.rl.compute_state_values_table(domain)

    # and store it
    with specmine.openz(out_path, "wb") as out_file:
        pickle.dump(values, out_file)

