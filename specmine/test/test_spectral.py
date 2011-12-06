import numpy
import nose.tools
import specmine

def assert_arrays_almost_equal(left, right, places = 7):
    left_list = left.round(places).tolist()
    right_list = right.round(places).tolist()

    nose.tools.assert_equal(left_list, right_list)

def test_laplacian_basis():
    adjacency_list = [
        [0, 1, 1, 0],
        [1, 0, 1, 1],
        [1, 1, 1, 0],
        [0, 1, 0, 1],
        ]
    adjacency_matrix = numpy.array(adjacency_list)
    basis_amg = specmine.spectral.laplacian_basis(adjacency_matrix, 3, method = "amg")
    basis_arpack = specmine.spectral.laplacian_basis(adjacency_matrix, 3, method = "arpack")
    basis_dense = specmine.spectral.laplacian_basis(adjacency_matrix, 3, method = "dense")

    print basis_amg
    print basis_arpack
    print basis_dense

    assert_arrays_almost_equal(numpy.abs(basis_amg), numpy.abs(basis_arpack))
    assert_arrays_almost_equal(numpy.abs(basis_arpack), numpy.abs(basis_dense))

