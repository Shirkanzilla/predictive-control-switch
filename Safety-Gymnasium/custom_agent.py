from omnisafe.models.actor import GaussianLearningActor
import safety_gymnasium
from safety_gymnasium.tasks.safe_navigation.goal.goal_level1 import GoalLevel1 
import torch
import numpy as np
from load_model import load_guide

import pdb
import os
os.environ["DISPLAY"] = ":0"

class LowFrictionPointGoal(GoalLevel1):
    def __init__(self, config):
        super().__init__(config=config)
        self.robot_model_file = os.path.abspath('custom_point.xml')



env_id = 'LowFrictionPointGoal1-v0'

config = {
    'agent_name': 'Point',
    'robot_model_file': 'custom_point.xml'
}

'''
safety_gymnasium.register(
    id=env_id,
    entry_point='custom_agent:LowFrictionPointGoal',
    kwargs={'config': config}
)
'''

env = safety_gymnasium.make("SafetyPointGoal1-v0", config=config, render_mode="human")
obs_space = env.observation_space
act_space = env.action_space

agent = GaussianLearningActor(obs_space, act_space, [4], activation='relu')
observation, info = env.reset()



episode_over = False
while not episode_over:
        obs_tensor = torch.from_numpy(observation).float()
        action = agent.predict(obs_tensor, deterministic=True).detach().numpy()
        observation, reward, cost, terminated, truncated, info = env.step(action)
        episode_over = terminated or truncated

env.close()