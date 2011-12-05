import cPickle as pickle
import numpy
import specmine
import condor

logger = specmine.get_logger(__name__)

def find_values(name, game):
    logger.info("computing values over game %s", name)

    (M, _) = game.moves.shape
    values = []

    for m in xrange(M):
        state = game.get_state(m)
        value = specmine.go.estimate_value(state, 32)

        values.append(value)

        logger.info("value is %.2f of:\n%s", value, state.board.grid)

    return numpy.array(values)

@specmine.annotations(
    workers = ("number of Condor workers", "option", "w", int),
    name = ("only analyze one game", "option"),
    )
def main(out_path, games_path, name = None, workers = 0):
    logger.info("reading games from %s", games_path)

    with specmine.util.openz(games_path) as games_file:
        games = pickle.load(games_file)

    if name is None:
        names = games
    else:
        names = [name]

    def yield_jobs():
        logger.info("distributing jobs for %i games", len(names))

        for name in names:
            yield (find_values, [name, games[name]])

    evaluated = []

    for (job, values) in condor.do(yield_jobs(), workers = workers):
        (game,) = job.args

        evaluated.append((game, values))

    with specmine.util.openz(out_path, "wb") as out_file:
        pickle.dump(evaluated, out_file, protocol = -1)

if __name__ == "__main__":
    specmine.script(main)

