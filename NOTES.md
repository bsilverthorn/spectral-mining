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

