import numpy
import sklearn.linear_model
import sklearn.cross_validation
import sklearn.metrics
import sklearn.pipeline
import specmine

logger = specmine.get_logger(__name__)

def evaluate_feature_map_rl(learn_policy, feature_map, games_for_testing = 500):
    """learns a policy using the learn_policy method and then tests against
    a random player, returning average reward"""

    # construct domain
    opponent_domain = specmine.rl.TicTacToeDomain(player = -1)
    opponent_policy = specmine.rl.RandomPolicy(opponent_domain)

    domain = specmine.rl.TicTacToeDomain(player = 1, opponent = opponent_policy)

    # learn a policy
    policy = learn_policy(domain, feature_map)

    # evaluate the policy
    rewards = []

    logger.info("evaluating TTT policy over %i games", games_for_testing)

    for i in xrange(games_for_testing):
        (s, r) = specmine.rl.generate_episode(domain, policy)

        rewards.append(r[-1])

    return (numpy.mean(rewards), numpy.var(rewards))

def evaluate_feature_map_td(feature_map, games_for_learning = 5000, games_for_testing = 500, **kwargs):
    def learn_td(domain, feature_map):
        return \
            specmine.rl.linear_td_learn_policy(
                domain,
                feature_map,
                games_for_learning,
                **kwargs)

    return evaluate_feature_map_rl(learn_td, feature_map, games_for_testing = games_for_testing)

def evaluate_feature_map_lstd(feature_map, games_per_eval = 1000, num_iters=10, games_for_testing = 500, **kwargs):
    def learn_lstd(domain, feature_map):
        return \
            specmine.rl.lstd_learn_policy(
                domain,
                feature_map,
                games_per_eval,
                num_iters,
                **kwargs)

    return evaluate_feature_map_rl(learn_lstd, feature_map, games_for_testing = games_for_testing)

def score_features_predict(feature_map, values, folds = 10, alpha = 0.1, append_affinity = True,**kwargs):
    """Score a feature map on value-function prediction."""

    # prepare features and targets
    states = values.keys()
    
    logger.info('getting the state values')
    
    state_values = numpy.array([values[s] for s in states]) # TODO can move this outside and pass in state values and boards

    logger.info('getting the feature values')

    state_features = numpy.array([feature_map[s] for s in states])    

    logger.info('appending the affinity representation')

    if append_affinity: # append the affinity representation to the feature vector
        aff_map = kwargs.get('affinity_map', specmine.feature_maps.flat_affinity_map)

        logger.info('using affinity map: %s',str(aff_map))

        aff_features = numpy.array(map(aff_map,states))

        state_features = numpy.hstack((numpy.ones((len(states),1)),aff_features, state_features))
    else:
        state_features = numpy.hstack((numpy.ones((len(states),1)),state_features)) # add constant to feature vector

    logger.info('setting up experiment')

    # run the experiment
    ridge = sklearn.linear_model.Ridge(alpha = alpha, normalize = True)
    regression = sklearn.pipeline.Pipeline([("scaler", sklearn.preprocessing.Scaler()), ("ridge",ridge),])

    logger.info('performing k fold cross val')

    k_fold_cv = sklearn.cross_validation.KFold(len(states), folds)
    scores = \
        sklearn.cross_validation.cross_val_score(
            regression,
            state_features,
            state_values,
            cv = k_fold_cv,
            score_func = rmse_score
            )

    return (numpy.mean(scores), numpy.var(scores))

def score_features_regress_act(
    feature_map,
    values,
    folds = 10,
    alpha = 1.0,
    games_for_testing = 1000,
    ):
    """Score a feature map on policy performance using least-squares weights."""

    # construct domain
    opponent_domain = specmine.rl.TicTacToeDomain(player = -1)
    opponent_policy = specmine.rl.RandomPolicy(opponent_domain)
    domain = specmine.rl.TicTacToeDomain(player = 1, opponent = opponent_policy)

    # prepare features and targets
    states = list(values)

    state_features = numpy.array([feature_map[s] for s in states])
    state_values = numpy.array([values[s] for s in states])

    # run the experiment
    ridge = sklearn.linear_model.Ridge(alpha = alpha)
    k_fold_cv = sklearn.cross_validation.KFold(len(states), folds)
    rewards = []

    for (train, _) in k_fold_cv:
        # learn a value function
        ridge.fit(state_features[train], state_values[train])

        value_function = specmine.rl.LinearValueFunction(feature_map, ridge.coef_)
        policy = specmine.rl.StateValueFunctionPolicy(domain, value_function)

        # evaluate the policy
        rewards = []

        logger.info("evaluating TTT policy over %i games", games_for_testing)

        for i in xrange(games_for_testing):
            (s, r) = specmine.rl.generate_episode(domain, policy)

            rewards.append(r[-1])

    # ...
    return (numpy.mean(rewards), numpy.var(rewards))

def rmse_score(values, prediction):
    ''' takes two numpy arrays and returns the RMSE between them'''
    error = (values-prediction)**2
    error = numpy.sqrt(numpy.mean(error))
    
    return error

