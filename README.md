Spectral Mining Notes
=====================

This package provides a loosely-organized set of tools for feature generation
using spectral graph theory on state spaces, primarily intended for value
function approximation in reinforcement learning.

Authors
-------

Craig Corcoran `<ccor@cs.utexas.edu>`
Bryan Silverthorn `<bcs@cargo-cult.org>`

Running Tests
-------------

`nosetests -v specmine --all-modules`

Modular RL Outline
------------------

State space pickle: `{state: [state]}`

"Domains" provide:

- `an initial_state`
- `states -> [state]`
- `outcome_of(state, action) -> state`
- `actions_in(state) -> [action]`
- `reward_in(state) -> scalar`
- `check_end(state) -> boolean`

Learning functions are of the form:

`learn(domain) -> policy`

Where policies are of the form:

`policy[state] -> action`
