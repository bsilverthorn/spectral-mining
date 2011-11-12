
class LinearValueFunction:
    def __init__(self,feature_map,weights):
        self.features = feature_map
        self.weights = weights

    def __getitem__(self,state):
        # TODO add interpolation to unseen states
        phi = self.features[state] # returns a vector of feature values
        return numpy.dot(phi,weights)

