# Documentation

Reinforcement learning loop:
* Initialize the environment
* Reset the environment to its initial state
* While the play is active (!(truncated or terminated)):
    * Agents take actions based on their observations
    * Actions are passed to the environment
    * Environment adjudicates actions, returns (observations, rewards, terminations, truncations, infos)
    * Agents update their decision-making algorithm
    * OPTIONAL: Log diagnostic/benchmarking information
* Repeat the above loop for large number of plays (until agents are well-trained)

The nature of our observation space and multi-agent learning means that traditional Q-Learning via a Q-Table is infeasible. While there are only 7,068 possible states for a single player, adding another player to the table increases the number of possible states up to 49,956,624. If a third entity is added as an observation (say the location of a target), the number of possible states grows to 3.5 * 10^11. 

## Code Structure

testbed.py
* Executes RL training loop
* Will likely be renamed in final version
* Testing ground during development

environments
* envs
    * multiagent_environment.py
        * Contains world state
        * Renders the world
        * Implements actions
        * Determine whether play has ended (termination/truncation)
    

## Scenarios

### DB Battle

DB Battle is the simplest scenario in this project, featuring two agents ("WR" and "DB") and a landmark on the field.
The WR's objective is to get to the landmark while creating separation from the DB, whereas the DB's objective is to stay as close as possible to the WR.
Ideally this generates emergent route-running behavior, as the DB doesn't know where the WR's landmark is, so the WR can attempt to "shake" the DB.
At the very least it provides a context to test all the functionality of the project before scaling up the complexity to a full game.

Agents:
* WR
    * Action Space: Discrete(4) -> {Right, Up, Left, Down}
    * Observation Space: Box([0, 0], [width - 1, height - 1], int)
        * Observations: Own location, DB location, landmark location
    * Reward Function: Dist(WR, DB) - Dist(WR, Landmark)
        * **Simple initial reward function, potentially improve to include a time penalty**
        * Receivers ideally reach their landmark within ~3s
* DB
    * Action Space: Discrete(4) -> {Right, Up, Left, Down}
    * Observation Space: Box([0, 0], [width - 1, height - 1], int)
        * Observations: Own location, WR location
    * Reward Function: -Dist(DB, WR)

### Full Game

Simulating a full American Football game is the ultimate goal of this project, initially on a grid field then with continuous motion & physics.
This scenario features 22 agents (11 offense, 11 defense), though the exact properties of these agents can vary based on user input.
A valid roster consists of 5 offensive linemen, 1 QB and some combination of WR,RB,TE for the remaining 5 offensive players.
Defensive positions will consist of DL, LB, CB, S, with no hard rules on what constitutes a valid set.

Action Spaces
* QB
    * ?Who to throw the ball to?
* Every other player:
    * Discrete(4) -> {Right, Up, Left, Down}