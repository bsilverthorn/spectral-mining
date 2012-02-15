import sys
import os
import csv
import re
import cPickle as pickle
import numpy
import sklearn.neighbors
import condor
import specmine


logger = specmine.get_logger(__name__)

@specmine.annotations(
    games_path = ("path to adjacency dict"),
    values_path = ("path to value function",),
    workers = ("number of condor jobs", "option", None, int),
    affinity_neighbors = ("number of neighbors used for affinity graph - controls sparsity", "option", None, int),
    interp_neighbors = ("number of neighbors to use during interpolation", "option", None, int),
    interp_sigma_sq = ("parameter controlling the width of gaussian neighbor weighting, if -1 flat average is used", "option", None, int),
    num_graph_samples = ("number of samples used for graph features", "option", None, int),
    num_test_samples = ("number of samples used for testing value prediction error", "option", None, int),
    max_num_features = ("max number of features used", "option", None, int),
    ridge_param = ("ridge regularization parameter", "option", None, float)
    )
def measure_feature_performance( \
    games_path, values_path,  workers = 0,\
    affinity_neighbors = 16, interp_neighbors = 8, interp_sigma_sq = 16,\
    num_graph_samples = 10000, num_test_samples = 20000, \
    max_num_features = 500, ridge_param = 0.05, feature_boost = True):
    
    value_player = ''
    if 'random' in values_path:
        value_player = 'random'
    elif 'alp' in values_path:
        value_player = 'alp'

    out_path = str.format('specmine/static/experiments/go_feature_performance.RMSE.{p}.rp={r}.ngs={g}.nts={t}.nf={f}.nan={an}.nin={inn}.is={sig}', \
                p=value_player, r = ridge_param, g=num_graph_samples, t=num_test_samples,f=max_num_features, \
                an=affinity_neighbors,inn=interp_neighbors, sig = interp_sigma_sq)
    out_path += '.boost.csv' if feature_boost else '.csv'

    logger.info('out path: %s', out_path)
    logger.info('interpolation sigma: %f', interp_sigma_sq)
 
    values = get_value_list(games_path,values_path)
    values = sorted(values, key = lambda _: numpy.random.rand()) # shuffle values


    full_value_dict = dict(values)
    sample_boards = full_value_dict.keys()[:num_graph_samples]

    # load or compute full feature maps
    full_2x2_temp_map = get_template_map(2, 2, B=numpy.inf, symmetric=True)
    if feature_boost: # use 2x2 template features as affinity map for building graph
        aff_map = full_2x2_temp_map
        full_laplacian_map = get_laplacian_map(sample_boards, num_samples = num_graph_samples, \
                max_eigs = max_num_features, neighbors=affinity_neighbors, affinity_map = aff_map)
        laplace_map_name = 'Boosted Laplacian'
    
    else:
        aff_map = specmine.feature_maps.flat_affinity_map
        full_laplacian_map = get_laplacian_map(sample_boards, num_samples = num_graph_samples, \
                max_eigs = max_num_features, neighbors=affinity_neighbors, affinity_map = aff_map)
        laplace_map_name = 'Laplacian'
        
    
    ball_tree = full_laplacian_map.ball_tree

    values = sorted(values, key = lambda _: numpy.random.rand()) # shuffle again before testing
    test_values = dict(values[:num_test_samples])

    def yield_jobs():
                
        logger.info("number of samples being used for graph features: %i", num_graph_samples)
        for NF in numpy.r_[0:max_num_features:10j].round().astype(int):
            yield (run_template_features, [test_values, full_2x2_temp_map, NF], dict(ridge_param = ridge_param))
            #yield (run_template_features, [full_3x3_temp_map, NF, test_values])
            yield (run_laplacian_features, [test_values, laplace_map_name, full_laplacian_map, NF, aff_map], dict(interp_neighbors = interp_neighbors, interp_sigma_sq = interp_sigma_sq, ridge_param = ridge_param))
            yield (run_random_features, [test_values, NF, ball_tree, aff_map], dict(interp_neighbors = interp_neighbors, interp_sigma_sq = interp_sigma_sq, ridge_param = ridge_param))

    with open(out_path, "wb") as out_file:
        writer = csv.writer(out_file)

        writer.writerow(["map_name", "features", "samples", "score_mean", "score_variance"])

        for (_, row) in condor.do(yield_jobs(), workers):
            writer.writerow(row)


def run_template_features(values, full_feature_map, B, symmetric=True, **kwargs):

    m,n = full_feature_map.grids[0].shape
    ridge_param = kwargs.get('ridge_param',1)

    logger.info("running %i by %i templates using %i features",m,n, B)

    feature_map = full_feature_map.gen_map(B)

    (mean, variance) = specmine.science.score_features_predict(feature_map, values, alpha = ridge_param)

    logger.info("with %i %s features, mean score is %f", B, 'template', mean)

    return [str(m)+'by'+str(n)+' template', feature_map.NF, None ,mean, variance]

def run_graph_features(values, map_name, B, features_NF, ball_tree, aff_map, **kwargs):
    interp_neighbors = kwargs.get('interp_neighbors',8)
    ridge_param = kwargs.get('ridge_param',1)

    feature_map = specmine.feature_maps.InterpolationFeatureMap(features_NF, aff_map, ball_tree, k=interp_neighbors)
    
    (mean, variance) = specmine.science.score_features_predict(feature_map, values, alpha = ridge_param)

    logger.info("with %i %s features, mean score is %f", B, map_name, mean)

    num_samples = ball_tree.data.shape[0]

    return [map_name, B, num_samples, mean, variance]


def run_laplacian_features(values, map_name, full_feature_map, B,  aff_map, **kwargs):

    logger.info("running laplacian features with %i eigenvectors", B)
    ball_tree = full_feature_map.ball_tree

    if B > 0:
        basis_NB = full_feature_map.basis[:,:B]

    else:
        basis_NB = None
        
    return run_graph_features(values, map_name, B, basis_NB, ball_tree, aff_map, **kwargs)


def run_random_features(values, B, ball_tree, aff_map, **kwargs):

    logger.info("running random features with %i eigenvectors", B)

    if B > 0:
        N = ball_tree.data.shape[0]
        features_NB = numpy.random.random((N, B))
    else:
        features_NB = None

    return run_graph_features(values, "random", B, features_NB, ball_tree,  aff_map, **kwargs)

def get_template_map(m, n, B=numpy.inf, symmetric=True):
    '''loads template map from file if available or generates the feature map. 
    If B is given as inf, then max number of features are used.'''
    if symmetric:
        path = str.format('specmine/static/feature_maps/template_feature_map.{a}x{b}.symmetric.pickle.gz',a=m,b=n)
    else:
        path = str.format('specmine/static/feature_maps/template_feature_map.{a}x{b}.pickle.gz',a=m,b=n)

    if os.path.isfile(path):
        #if available, use precomputed feature map
        logger.info("using precomputed features at %s", path)
        with specmine.openz(path) as featuremap_file:
            full_feature_map = pickle.load(featuremap_file)
    else:
        # generate and save for next time
        logger.info("generating complete %i by %i feature map",m,n)

        full_feature_map = specmine.feature_maps.TemplateFeatureMap(m,n)
        
        logger.info('saving computed feature map: %s', path)
        with specmine.util.openz(path, "wb") as out_file:
            pickle.dump(full_feature_map, out_file)

    if B == numpy.inf:
        return full_feature_map
    else:
        return full_feature_map.gen_map(B)

def get_laplacian_map(boards=None, num_samples=10000, max_eigs=500, neighbors=8, \
                        affinity_map = specmine.feature_maps.flat_affinity_map):

    root,dirs,files = os.walk('./specmine/static/feature_maps/').next()
    curr_dir = 'specmine/static/feature_maps/'

    if affinity_map == specmine.feature_maps.flat_affinity_map:
        aff = 'flat'
    elif type(affinity_map) == specmine.feature_maps.TemplateFeatureMap:
        logger.info('getting laplacian map using template features as affinity')
        aff = '2x2_sym_template'
        affinity_map = affinity_map.__getitem__
    else:
        aff= '?'

    path_front = str.format('laplacian.ngs={s}.nan={k}.aff={a}',s=num_samples, k=neighbors, a=aff)
    path_end = '.pickle.gz'
    
    precomp = False

    for f in files:

        match = re.search(path_front+'.nf=(\d+)'+path_end, f)
        
        if match is not None:
            
            path = curr_dir + match.group(0)
            num_feats = int(match.group(1))
            
            if num_feats >= max_eigs:
                
                logger.info("using precomputed features at %s", path)
                logger.info('num features used: %i , max available: %i',num_feats,max_eigs)

                with specmine.openz(path) as featuremap_file:
                    full_feature_map = pickle.load(featuremap_file)
                    
                precomp = True
                break


    if not precomp:

        path = curr_dir + path_front + str.format('.nf={n}', n= max_eigs) + path_end
        # generate and save for next time
        logger.info("generating laplacian eigenvector feature map with %i eigenvectors",max_eigs)
        
        avectors_ND = numpy.array(map(affinity_map, boards))
        affinity_NN, ball_tree = specmine.feature_maps.build_affinity_graph(avectors_ND, neighbors, get_tree=True)

        basis_NB = specmine.spectral.laplacian_basis(affinity_NN, max_eigs, method = "arpack") # amg

        full_feature_map = specmine.feature_maps.InterpolationFeatureMap(basis_NB, \
                                specmine.feature_maps.flat_affinity_map,
                                ball_tree)

        logger.info('saving computed laplacian feature map: %s', path)
        with specmine.util.openz(path, "wb") as out_file:
            pickle.dump(full_feature_map, out_file)
        

    return full_feature_map


def get_value_list(games_path,values_path):
    ''' takes a list of games and a dictionary of values and builds a list of
    (BoardState, value) pairs '''

    with specmine.util.openz(games_path) as games_file:
        games = pickle.load(games_file)

    with specmine.util.openz(values_path) as values_file:
        values = pickle.load(values_file)

    value_list = []

    for game in values.keys():
        try:
            vals = values[game]
            boards = list(set(map(specmine.go.BoardState, games[game].grids)))
            value_list.extend(zip(boards,vals))
        except KeyError:
            print 'game unkown for ',game

    # remove duplicates
    value_dict = dict(value_list) 
    value_keys = value_dict.keys()
    value_list = zip(value_keys,map(value_dict.__getitem__,value_keys))

    return value_list

if __name__ == "__main__":
    specmine.script(measure_feature_performance)
