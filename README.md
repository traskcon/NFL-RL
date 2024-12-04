# NFL-RL
Training agents to play American Football via Reinforcement Learning

_Please report any bugs via this repository's "Issues" section_

## Installation Instructions

All relevant code files can be found at this project's GitHub Repository: https://github.com/traskcon/NFL-RL
Development notes and descriptions of scenarios, code flows, and code structure can be found in DOCS.md

### Initial Setup
1. Download code files: ``git clone https://github.com/traskcon/NFL-RL``
2. Navigate inside the NFL-RL directory and install all necessary packages:
    ``cd NFL-RL``
    ``pip install -r requirements.txt``

### Run DB Battle Scenario
1. To run the DB Battle Scenario using the trained agents, run ``python demonstration.py --scenario="db-battle" --method="DQN"``
   NOTE: The arguments are case-sensitive, therefore copy-pasting the above line into the command line is preferred
2. Alternatively, one can run the DB Battle Scenario with the agents taking the action with the greatest immediate reward using ``python demonstration.py --scenario="db-battle" --method="heuristic"``. This is useful for demonstrating the ability of the DQNs to learn long-term behavior

**Expected Result:** A pygame window will open, showing the WR (blue) running towards their target location (white square), while being closely shadowed by the DB (red). The simulation will end once the WR reaches the target location. Notably, the target location present used in this demonstration was not present in the training set for these agents. 

### Run All-22 Scenario
1. Similarly to the DB Battle Scenario, the All-22 Scenario can be run with trained agents via ``python demonstration.py --scenario="all-22" --method="DQN"``
2. To run with agents taking the action with the greatest immediate reward: ``python demonstration.py --scenario="all-22" --method="heuristic"``

**Expected Result:** A pygrame window will open, showing both teams lined up on either side of the line of scrimmage (blue line). The blue receivers will begin running routes, while the offensive line will push against the red defensive line. Since passing and handoffs have not yet been implemented, the QB (yellow circle) will shuffle around the pocket, until eventually a red defender closes the distance and tackles them, causing the simulation to end.

## Development Roadmap

This project will be built using the PettingZoo environment for multi-agent RL simulations, 
however instead of following their modular scenario approach, the majority of the environmental logic will be built directly into the environment itself.
Additional features will be offloaded to their own classes where it makes sense to do so, but this will likely follow after direct implementation.

### Proof-of-Concept
 * ~~Create 2D, rectangular grid environment~~ (DONE)
 * ~~Build test agent~~ (DONE)
 * ~~Implement basic reward function~~ (DONE)
 * ~~Visualize results~~ (DONE)

### Initial Prototype
 * Scale-up the number of agents present
    * ~~Begin with adding a "DB" agent who shadows "WR" agent~~
    * Then scale to 11 agent offense
    * Finally full 22 agent game
 * Refine environment
    * Add logic checks for endzones, sidelines
 * Implement football-specific reward functions
    * Discrete Rewards (Occur at the end of a play):
        * EPA from the play (+Off, -Def)
        * TD (+Off, -Def)
        * INT, FUM (-Off, +Def)
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