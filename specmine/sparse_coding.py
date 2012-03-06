import pickle
import numpy
import specmine
#import pyplot.matplotlib as pl
import pylab as pl
from time import time
from sklearn.decomposition import MiniBatchDictionaryLearning

def main(games_path = None):
    
    if games_path == None:
        games_path = 'specmine/data/go_games/2010-01.pickle.gz'

    with specmine.util.openz(games_path) as games_file:
        games = pickle.load(games_file)

    boards = None # numpy array nx9x9 
    for game in games:
        if boards == None: 
            boards = games[game].grids
        else:
            boards = numpy.vstack((boards,games[game].grids))

    print 'boards shape: ', boards.shape

    print 'Learning the dictionary... '
    t0 = time()
    dico = MiniBatchDictionaryLearning(n_atoms=100, alpha=1, n_iter=500)
    V = dico.fit(boards).components_
    dt = time() - t0
    print 'done in %.2fs.' % dt

    pl.figure(figsize=(4.2, 4))
    for i, comp in enumerate(V[:100]):
        pl.subplot(10, 10, i + 1)
        pl.imshow(comp.reshape(patch_size), cmap=pl.cm.gray_r,
              interpolation='nearest')
        pl.xticks(())
        pl.yticks(())
        pl.suptitle('Dictionary learned from Lena patches\n' +
            'Train time %.1fs on %d patches' % (dt, len(data)),
            fontsize=16)
        pl.subplots_adjust(0.08, 0.02, 0.92, 0.85, 0.08, 0.23)

if __name__ == '__main__':
    main()
