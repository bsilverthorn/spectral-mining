import numpy

def applyTemplates(applications, templates, state, num_feats):

    TR = templates[0].shape[0]
    TC = templates[0].shape[1]
    counts = numpy.zeros(num_feats, numpy.uint32)

    for i in xrange(applications.shape[0]):
        r = applications[i,0]
        c = applications[i,1]
        t = applications[i,2]
        f = applications[i,3]

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
