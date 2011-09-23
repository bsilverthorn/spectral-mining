import plac
import numpy
import spectral

@plac.annotations(
    num_app = ("application count", "positional", None, int),
    )
def main(num_app = 10):
    size = 10 
    W = spectral.room_adjacency(size)
    L = spectral.laplacian_operator(W)
    #L = spectral.diffusion_operator(W)
    V = numpy.zeros(W.shape[0])

    V[42] = 1.0

    U = numpy.empty((W.shape[0], num_app))
    U[:, 0] = V

    for i in xrange(1, num_app):
        U[:, i] = numpy.dot(L, U[:, i - 1])

    spectral.plot_functions_on_room(size, U[:,-10:])

if __name__ == "__main__":
    plac.call(main)

