import tarfile
import specmine

logger = specmine.get_logger(__name__)

def main(*archive_files):
    games = []
    games_seen = 0
    boards = set()
    boards_seen = 0

    for af in archive_files:
        logger.info('opening archive: %s', af)

        archive = tarfile.open(af)
        names = [s for s in archive.getnames() if s.endswith(".sgf")]

        for name in names:
            logger.info('playing game: %s', name)

            f = archive.extractfile(name)
            s,r = specmine.go.read_expert_episode(f)

            games_seen += 1
            if len(s) > 0:
                games.append((s, r))

            boards_seen += len(s)
            boards.update(s)

            logger.info("taking %i games of %i seen", len(games), games_seen)
            logger.info(
                "of %i boards seen, %i are unique (%.2f%%)",
                boards_seen,
                len(boards),
                100.0 * len(boards) / (float(boards_seen) + 1e-16),
                )

            f.close()

if __name__ == "__main__":
    specmine.script(main)

