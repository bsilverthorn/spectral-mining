Spectral Mining Notes
=====================

This package provides a loosely-organized set of tools for feature generation
using spectral graph theory on state spaces, primarily intended for value
function approximation in reinforcement learning.

Authors
-------

Craig Corcoran `<ccor@cs.utexas.edu>`
Bryan Silverthorn `<bcs@cargo-cult.org>`

Installation
------------

This package is implemented in Python. To begin, set up a virtual environment
with virtualenv. From the project root (the directory containing this README):

`python virtualenv.py --no-site-packages environment`
`source environment/bin/activate`

Then install the minimum set of relevant dependencies:

`easy_install numpy`
`easy_install scipy`
`easy_install scikit-learn`
`easy_install plac`

The project experiments are organized as modules under the
"specmine.experiments" package. To obtain, for example, the Tic-Tac-Toe
eigenvector visualization data, run

`python -m specmine.experiments.ttt_analyze_evs eigenvectors.csv`

# XXX configure the static symlink

