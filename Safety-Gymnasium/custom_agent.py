from omnisafe.models.actor import GaussianLearningActor
import safety_gymnasium
import torch
import numpy as np
from math import sqrt
from load_model import load_guide
from safety_gymnasium.assets.geoms import Hazards
from safety_gymnasium.assets.geoms import Sigwalls
from safety_gymnasium.bases.base_task import BaseTask
from safety_gymnasium.utils.registration import register

import pdb
import os
os.environ["DISPLAY"] = ":0"

env_id = "SafetyPointSigWallGoalLevel1-v0"
config = {'agent_name': 'Point'}
kwargs = {'config': config, 'task_id': env_id}
register(id=env_id, entry_point='omnisafe.envs.sagui_builder:SaguiBuilder', kwargs=kwargs, max_episode_steps=1000)

coefs = None

def set_coef_dict(coef_dict: dict):
    global coefs  # bad practice but ok
    coefs = coef_dict

def _modify_dyn(model, coef_dict: dict):
    for name, mult in coef_dict.items():
        atr: np.ndarray = getattr(model, name)
        atr[:] *= mult

def _set_default_dyn(model):
    model.dof_damping[0] *= 1.5  # Axis X
    model.dof_damping[1] *= 1.5  # Axis Z
    # model.dof_damping[2] *= 1.0  # Steering

    if coefs != None:
        _modify_dyn(model, coefs)


class SigWallGoalLevel1(BaseTask):
    """An agent must navigate to a goal."""

    def __init__(self, config) -> None:
        super().__init__(config=config)

        self._add_geoms(Hazards(num=1, keepout=0.75, size=0.7, locations=[(0, 0)]))
        self._add_geoms(Sigwalls(num=4, locate_factor=2.5, is_constrained=True))

        self.placements_conf.extents = [-1.75, -1.75, 1.75, 1.75]

    def calculate_reward(self):
        x0, y0, _ = self.last_robot_pos
        x, y, _ = self.agent.pos
        reward = sqrt((x - x0)**2 + (y - y0)**2)

        return reward

    def specific_reset(self):
        self.last_robot_pos = self.agent.pos
        _set_default_dyn(self.model)

    def specific_step(self):
        self.last_robot_pos = self.agent.pos

    def update_world(self):
        self.last_robot_pos = self.agent.pos

    @property
    def goal_achieved(self):
        """Whether the goal of task is achieved."""
        # pylint: disable-next=no-member
        return self.dist_goal() <= self.goal.size



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