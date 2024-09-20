# Documentation

Reinforcement learning loop:
* Initialize the environment
* Reset the environment to its initial state
* While the play is active (!(truncated or terminated)):
    * Agents take actions
    * Actions are passed to the environment
    * Environment adjudicates actions, returns (observations, rewards, terminations, truncations, infos)
    * Agents update their decision-making algorithm
    * OPTIONAL: Log diagnostic/benchmarking information
* Repeat the above loop for large number of plays (until agents are well-trained)


