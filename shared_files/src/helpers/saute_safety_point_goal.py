from __future__ import annotations

import os
import random
import omnisafe
from typing import Any, ClassVar

import torch
from gymnasium import spaces

from omnisafe.envs.core import CMDP, env_register, env_unregister
from omnisafe.typing import DEVICE_CPU, Box

import numpy as np

from gymnasium import utils
from gymnasium.envs.mujoco import MujocoEnv
from gymnasium.spaces import Box

import omnisafe
from omnisafe.envs.core import env_register

import safety_gymnasium
from safety_gymnasium.assets.free_geoms import Vases
from safety_gymnasium.assets.geoms import Hazards
from safety_gymnasium.tasks.safe_navigation.goal.goal_level0 import GoalLevel0

dir_path = os.path.dirname(os.path.realpath(__file__))

class SauteGoalLevel1(GoalLevel0):
    """An agent must navigate to a goal while avoiding hazards.

    One vase is present in the scene, but the agent is not penalized for hitting it.
    """

    def __init__(self, cost_budget, gamma, config) -> None:
        super().__init__(config=config)
        self.placements_conf.extents = [-1.5, -1.5, 1.5, 1.5]

        self._add_geoms(Hazards(num=8, keepout=0.18))
        self._add_free_geoms(Vases(num=1, is_constrained=False))
        self._cost_budget = cost_budget
        self._inital_cost_budget = cost_budget
        self._gamma = gamma

    def calculate_reward(self):
        reward = super().calculate_reward()
        self._cost_budget = (self._cost_budget - self.calculate_cost()) / self._gamma
        if self._cost_budget < 0:
            reward = - 1000000.0
        return reward

    def obs(self):
        obs = super().obs()
        obs['normalized_cost_budget'] = self._cost_budget / self._inital_cost_budget
        return obs
    
class OmnisafeSauteGoalLevel1(CMDP):
    _support_envs: ClassVar[list[str]] = ["SautePointGoal1-v0"]

    need_auto_reset_wrapper = False
    need_time_limit_wrapper = False

    def __init__(self, env_id: str, num_envs: int, device: torch.device = DEVICE_CPU, **kwargs) -> None:
        self._count = 0
        self._num_envs = 1 # not supported
        self._inner_env = safety_gymnasium.make(id="SautePointGoal1-v0", autoreset=False, **kwargs)
        self._observation_space = self._inner_env.observation_space
        self._action_space = self._inner_env.action_space
        self._device = torch.device(device)

    def set_seed(self, seed: int) -> None:
        random.seed(seed)

    def reset(
        self,
        seed: int | None = None,
        options: dict[str, Any] | None = None,
        ) -> tuple[torch.Tensor, dict]:

        obs, info = self._inner_env.reset(seed=seed, options=options)
        return torch.as_tensor(obs, dtype=torch.float32, device=self._device), info

    @property
    def max_episode_steps(self) -> None:
        """The max steps per episode."""
        return self._inner_env.spec.max_episode_steps
    
    def render(self) -> Any:
        pass

    def close(self) -> Any:
        self._inner_env.close()

    def step(
        self,
        action: torch.Tensor,
        ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, dict]:
        obs, reward, cost, terminated, truncated, info = self._inner_env.step(
            action.detach().cpu().numpy(),
        )
        obs, reward, cost, terminated, truncated = (
            torch.as_tensor(x, dtype=torch.float32, device=self._device)
            for x in (obs, reward, cost, terminated, truncated)
        )
        if 'final_observation' not in info:
            info['final_observation'] = obs
        info['final_observation'] = np.array(
            [
                array if array is not None else np.zeros(obs.shape[-1])
                for array in info['final_observation']
            ],
        )
        info['final_observation'] = torch.as_tensor(
            info['final_observation'],
            dtype=torch.float32,
            device=self._device,
        )

        return obs, reward, cost, terminated, truncated, info
