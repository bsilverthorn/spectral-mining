# cython: profile=True
import numpy

cimport cython
cimport numpy

@cython.boundscheck(False)
def applyTemplates(py_applications, py_templates, py_state, num_feats):
    cdef numpy.ndarray[numpy.uint32_t, ndim = 2] applications = py_applications.astype(numpy.uint32)
    cdef numpy.ndarray[numpy.uint8_t, ndim = 3] templates = py_templates.astype(numpy.uint8)
    cdef numpy.ndarray[numpy.uint8_t, ndim = 2] state = py_state.astype(numpy.uint8)
    cdef numpy.ndarray[numpy.uint32_t, ndim = 1] counts = numpy.zeros(num_feats, numpy.uint32)

    cdef unsigned int i
    cdef unsigned int r
    cdef unsigned int c
    cdef unsigned int t
    cdef unsigned int tr
    cdef unsigned int tc
    cdef unsigned int TR = templates.shape[1]
    cdef unsigned int TC = templates.shape[2]

    for i in xrange(applications.shape[0]):
        r = applications[i, <unsigned int>0]
        c = applications[i, <unsigned int>1]
        t = applications[i, <unsigned int>2]
        f = applications[i, <unsigned int>3]

        match = True

        for tr in xrange(TR):
            for tc in xrange(TC):
                if templates[t, tr, tc] != state[r + tr, c + tc]:
                    match = False

                    break

            if not match:
                break

        if match:
            counts[f] += 1

    ind = counts.nonzero()[0] #  weight indices where the number of features present was nonzero
    counts = counts[ind]

    return (ind, counts)

