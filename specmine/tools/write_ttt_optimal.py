import specmine.tools.write_ttt_optimal

if __name__ == "__main__":
    specmine.script(specmine.tools.write_ttt_optimal.main)

import cPickle as pickle
import specmine

logger = specmine.get_logger(__name__)

specmine.annotations(
    out_path = ("path to write policy pickle",),
    player = ("player number", "option", None, int),
    )
def main(out_path, player = -1):
    """Compute the optimal TTT policy for the specified player."""

    # construct domain
    opponent_domain = specmine.rl.TicTacToeDomain(player = -1 * player)
    opponent_policy = specmine.rl.RandomPolicy(opponent_domain)

    domain = specmine.rl.TicTacToeDomain(player = player, opponent = opponent_policy)

    # compute the optimal policy
    policy = {}

    for state in domain.states:
        (board, player_to_move) = state

        if player == player_to_move:
            (move_i, move_j, _) = specmine.tictac.ab_optimal_move(board, player)

            move = (move_i, move_j)

            policy[state] = move
        else:
            actions = list(domain.actions_in(state))

            if actions:
                (move,) = actions

                policy[state] = move
            else:
                move = "(Terminal)"

        logger.info("optimal move in %s: %s", board._grid.astype(int).tolist(), move)

    # and store it
    with specmine.openz(out_path, "wb") as out_file:
        pickle.dump(policy, out_file)

