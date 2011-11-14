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

