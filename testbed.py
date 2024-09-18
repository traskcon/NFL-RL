import gymnasium as gym
import environments

# Load environment
env = gym.make("environments/GridField-v0", render_mode="human")
observation, info = env.reset() #Initialize environment using reset method

for _ in range(1000):
    action = env.action_space.sample() #Randomly sample from the environment's action space
    observation, reward, terminated, truncated, info = env.step(action) #Return the results of that action

    if terminated or truncated:
        observation, info = env.reset()

env.close()
