Design of a Naïve RL Framework
==============================

General Notes
-------------

State space pickle scheme: `{state: [state]}`

RL Scheme
---------

Domains provide methods:

- `states -> [state]`
- `outcome_of(state, action) -> state`
- `actions_in(state) -> [action]`
- `reward_in(state) -> scalar`

Learning functions are of the form:

`learn(domain) -> policy`

Where policies are of the form:

`policy[state] -> action`

