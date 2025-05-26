from omnisafe.models.actor import GaussianLearningActor
import safety_gymnasium
import torch
import numpy as np
from load_model import load_guide


env = safety_gymnasium.make('SafetyPointGoal1-v0', render_mode="human")
#env = safety_gymnasium.make('SafetyCarFormulaOne1-v0', render_mode="human")
obs_space = env.observation_space
act_space = env.action_space

agent = GaussianLearningActor(obs_space, act_space, [256,256,256,256,256], activation='relu', weight_initialization_mode="orthogonal")
safe_agent = load_guide("/home/user/bachelor/runs/PPOLag-{SafetyPointGoal1-v0}/seed-000-2025-05-13-17-51-08", "epoch-50.pt")[1]
observation, info = env.reset()

episode_over = False
is_sampling = False
cost_window=200
probability_for_datapoint = 0.02
sampling_step = 0
while not episode_over:
    if is_sampling:
        obs_tensor = torch.from_numpy(observation).float()
        action = safe_agent.predict(obs_tensor, deterministic=True).detach().numpy()
        observation, reward, cost, terminated, truncated, info = env.step(action)
        episode_over = terminated or truncated
        sampling_step += 1
        if sampling_step >= cost_window or episode_over:
            is_sampling = False
            sampling_step = 0
            print("unsafe")
    else:
        print(env.unwrapped.__getattribute__("agent").model.body_pos)
        obs_tensor = torch.from_numpy(observation).float()
        action = agent.predict(obs_tensor, deterministic=True).detach().numpy()
        if np.random.random()<probability_for_datapoint:
            # begin sampling with the safe agent
            is_sampling = False
            print("safe")
        observation, reward, cost, terminated, truncated, info = env.step(action)
        episode_over = terminated or truncated

env.close()