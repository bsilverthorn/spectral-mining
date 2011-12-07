import cPickle as pickle
import tarfile
import contextlib
import specmine

logger = specmine.get_logger(__name__)

def read_sgf_archive(archive_path):
    logger.info("opening archive %s", archive_path)

    games = {}

    with contextlib.closing(tarfile.open(archive_path)) as archive:
        names = [s for s in archive.getnames() if s.endswith(".sgf")]

        for (n, name) in enumerate(names):
            logger.info("extracting game %s (%i of %i); %i retained", name, n + 1, len(names), len(games))

            with contextlib.closing(archive.extractfile(name)) as game_file:
                game = specmine.go.read_sgf_game(game_file)

                if game is not None:
                    games[name] = game

    logger.info("finished extracting games from %s", archive_path)

    return games

@specmine.annotations(
    track_boards = ("track unique board count?", "flag"),
    )
def main(out_path, track_boards = False, *archive_paths):
    """Extract games from SGF archives."""

    games = {}

    for archive_path in archive_paths:
        these = read_sgf_archive(archive_path)

        games.update(these)

    logger.info("pickling %i games to %s", len(games), out_path)

    with specmine.util.openz(out_path, "wb") as out_file:
        pickle.dump(games, out_file, protocol = -1)

if __name__ == "__main__":
    specmine.script(main)

