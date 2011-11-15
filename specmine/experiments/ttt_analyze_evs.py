import specmine.experiments.ttt_analyze_evs

if __name__ == "__main__":
    specmine.script(specmine.experiments.ttt_analyze_evs.main)

import csv
import specmine

logger = specmine.get_logger(__name__)

specmine.annotations(
    out_path = ("path to write output CSV",),
    number = ("number of eigenvectors to compute", "option", None, int),
    )
def main(out_path, number = 9):
    """Analyze eigenvectors in the TTT domain."""

    B = number
    adict = specmine.tictac.load_adjacency_dict()
    (gameplay_NN, index) = specmine.discovery.adjacency_dict_to_matrix(adict)
    basis_NB = specmine.spectral.laplacian_basis(gameplay_NN, B)
    start = specmine.tictac.BoardState()
    rows = []

    for i in xrange(3):
        for j in xrange(3):
            board = start.make_move(1, i, j)
            n = index[(board, -1)]

            for b in xrange(B):
                rows.append([b, i, j, basis_NB[n, b]])

    with specmine.openz(out_path, "wb") as out_file:
        writer = csv.writer(out_file)

        writer.writerow(["number", "i", "j", "value"])
        writer.writerows(rows)

