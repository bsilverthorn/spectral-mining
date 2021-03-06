import cPickle as pickle
import random
import numpy
import specmine
import condor

logger = specmine.get_logger(__name__)

def find_values(name, game, rollouts, winrate=True):
    (M, _) = game.moves.shape
    values = []

    logger.info("evaluating all %i positions in game %s", M, name)

    for m in xrange(M):

        board = specmine.go.replay_moves(game.moves[:m + 1])
       
        value = specmine.go.estimate_value(board, rollouts, \
            player = specmine.go.FuegoRandomPlayer(board), \
            opponent = specmine.go.FuegoRandomPlayer(board), \
            winrate = winrate)
        #value = specmine.go.estimate_value(board, rollouts, \
            #winrate = winrate)

        values.append(value)

        logger.info("%s grid %i, below, has value %f:\n%s", name, m, value, game.grids[m])
        
    return numpy.array(values)

@specmine.annotations(
    workers = ("number of Condor workers", "option", "w", int),
    name = ("only analyze one game", "option"),
    samples = ("number of games to sample", "option", None, int),
    rollouts = ("rollouts to perform", "option", None, int),
    )
def main(out_path, games_path, name = None, samples = None, rollouts = 256, workers = 0):
    logger.info("reading games from %s", games_path)

    with specmine.util.openz(games_path) as games_file:
        games = pickle.load(games_file)

    if name is None:
        if samples is None:
            names = games
        else:
            names = sorted(games, key = lambda _: random.random())[:samples]
    else:
        names = [name]

    def yield_jobs():
        logger.info("distributing jobs for %i games", len(names))

        for name in names:
            yield (find_values, [name, games[name], rollouts])

    evaluated = {}

    for (job, values) in condor.do(yield_jobs(), workers = workers):
        (name, _, _) = job.args

        evaluated[name] = values

    with specmine.util.openz(out_path, "wb") as out_file:
        pickle.dump(evaluated, out_file, protocol = -1)

if __name__ == "__main__":
    specmine.script(main)

