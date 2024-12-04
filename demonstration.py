# Script to demonstrate trained agents operating in the environment
from environments.envs import multiagent_environment
from environments import db_battle, all_22
import argparse
import sys
import numpy as np
import policy

parser = argparse.ArgumentParser(description="Run a single GRIDIRON Simulation")
parser.add_argument("--scenario", required=True, type=str)
parser.add_argument("--method", required=True, type=str)
args = parser.parse_args()

scenario_name = args.scenario
method = args.method

if scenario_name == "db-battle":
    scenario = db_battle.Scenario()
    roster = "DB-Battle-Roster.csv"
    model_prefix = "-MK3_MR"
elif scenario_name == "all-22":
    scenario = all_22.Scenario()
    roster = "FullTeam-Roster.csv"
    model_prefix = "-A22-MK1"
else:
    sys.exit("Invalid Scenario Entered")

env = multiagent_environment.MultiEnvironment(scenario=scenario, max_cycles=100, render_mode="human", roster=roster)
observations = env.reset()
learner = policy.Policy(env, observations)
learner.load_models(model_prefix)

epsilon = 0.01

while env.world.agents:
    if np.random.random() <= epsilon:
        actions = {agent.name: env.action_space(agent).sample() for agent in env.world.agents}
    else:
        actions = {agent.name: learner.choose_action(agent, observations[agent.name], method=method)
            for agent in env.world.agents}
    new_observations, rewards, terminations, truncations = env.step(actions)
    env.render()
    observations = new_observations
env.close()