import cPickle as pickle
import tarfile
import specmine

logger = specmine.get_logger(__name__)

@specmine.annotations(
    track_boards = ("track unique board count?", "flag"),
    )
def main(out_path, track_boards = False, *archive_paths):
    games = []
    boards = set()
    boards_seen = 0

    for archive_path in archive_paths:
        logger.info('opening archive: %s', archive_path)

        archive = tarfile.open(archive_path)
        names = [s for s in archive.getnames() if s.endswith(".sgf")]

        for (n, name) in enumerate(names):
            logger.info('playing game %s (%i of %i)', name, n + 1, len(names))

            f = archive.extractfile(name)
            s,r = specmine.go.read_expert_episode(f)

            if len(s) > 0:
                boards_seen += len(s)

                games.append((s, r))

            logger.info("stored %i games and %i boards", len(games), boards_seen)

            if track_boards:
                boards.update(s)

                logger.info(
                    "of %i boards seen, %i are unique (%.2f%%)",
                    boards_seen,
                    len(boards),
                    100.0 * len(boards) / (float(boards_seen) + 1e-16),
                    )

            f.close()

        archive.close()

    logger.info("pickling %i games to %s", len(games), out_path)

    with specmine.util.openz(out_path, "wb") as out_file:
        pickle.dump(games, out_file, protocol = -1)

if __name__ == "__main__":
    specmine.script(main)

