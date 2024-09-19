# NFL-RL
Training agents to play American Football via Reinforcement Learning

## Development Roadmap

This project will be built using the PettingZoo environment for multi-agent RL simulations, 
however instead of following their modular scenario approach, the majority of the environmental logic will be built directly into the environment itself.
Additional features will be offloaded to their own classes where it makes sense to do so, but this will likely follow after direct implementation.

### Proof-of-Concept
 * ~~Create 2D, rectangular grid environment~~ (DONE)
 * Build test agent
 * Implement basic reward function
 * Visualize results

### Initial Prototype
 * Scale-up the number of agents present
    * Begin with adding a "DB" agent who shadows "WR" agent
    * Then scale to 11 agent offense
    * Finally full 22 agent game
 * Refine environment
    * Add logic checks for endzones, sidelines
 * Implement football-specific reward functions
    * Discrete Rewards (Occur at the end of a play):
        * EPA from the play (+Off, -Def)
        * TD (+Off, -Def)
        * INT (-Off, +Def)
    * Continuous Rewards (Occur each tick):
        * For WRs: Yds of separation (euclidean distance from nearest DB)
        * For DBs: -Yds of separation
        * For OL: Pass protection (euclidean distance from nearest DL to QB)
        * For QB: Pocket Presence (euclidean distance from nearest defender)
        * ...
 * Add football-specific features and agent logic
    * QB agent needs a process for deciding who to throw to, or if they should scramble
    * "Catching" stat for WRs and DBs to determine if a ball is caught (np.random.rand <= catching)
    * "Strength" stat for all agents to determine who succeeds when attempting to move against each other
    * ...
 * Train model and observe initial results

### Future Refinements
 * Create NFL-representative rosters of agents
    * Combination of past performance and physical characteristics into determining stats
 * Convert grid environment into a continuous environment
    * Agents still represented with a simple shape (square, circle)
    * Movement becomes continuous, necessitating "speed" stats for all agents
        * Contested movement also becomes continuous, instead of winner-takes-all
 * Prompt the agents via playcall
    * Unsure exactly how this will be implemented (change decision-making process? reward function?)
    * Essentially refine the agent's objectives
        * Still trying to block, get open, etc., but within the context of running a specific route, blocking scheme, etc.