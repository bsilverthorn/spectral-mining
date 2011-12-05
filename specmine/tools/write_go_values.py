import cPickle as pickle
import specmine
import condor

logger = specmine.get_logger(__name__)

def find_value(game_state):
    return specmine.go.estimate_value(game_state, 32)

@specmine.annotations(
    workers = ("number of Condor workers", "option", "w", int),
    )
def main(out_path, games_path, workers = 0):
    logger.info("reading games from %s", games_path)

    with specmine.util.openz(games_path) as games_file:
        games = pickle.load(games_file)

    def yield_jobs():
        logger.info("distributing jobs for %i games", len(games))

        for (states, rewards) in games:
            for state in states:
                yield (find_value, [state])

    values_dict = {}

    for (job, value) in condor.do(yield_jobs(), workers = workers):
        (state,) = job.args

        values_dict[state] = value

    with specmine.util.openz(out_path, "wb") as out_file:
        pickle.dump(values_dict, out_file, protocol = -1)

if __name__ == "__main__":
    specmine.script(main)

