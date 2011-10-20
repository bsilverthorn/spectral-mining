Design of a Naïve RL Framework
==============================

Domains provide:

- `states -> [state]`
- `outcome_of(state, action) -> state`
- `actions_in(state) -> [action]`
- `reward_in(state) -> scalar`

Learning methods are of the form:

`learn(domain) -> policy`

Where policies are of the form:

`policy[state] -> action`

