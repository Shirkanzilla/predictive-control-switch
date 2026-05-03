from __future__ import annotations

import os
import random
import omnisafe
from typing import Any, ClassVar

import torch
from gymnasium import spaces

from omnisafe.envs.core import CMDP, env_register, env_unregister
from omnisafe.typing import DEVICE_CPU, Box
from sklearn.base import BaseEstimator

import numpy as np

import safety_gymnasium
from gymnasium import utils
from gymnasium.envs.mujoco import MujocoEnv
from gymnasium.spaces import Box

import omnisafe
from omnisafe.envs.core import env_register
from helpers.saute_omnisafe_inverted_pendulum import SauteInvertedPendulumEnv
from helpers.saute_safety_gym_envs import SauteSafetyGymEnv

class NeuralShieldingEnv(CMDP):
    _support_envs: ClassVar[list[str]] = ["NeuralShieldingInvertedPendulum-v4", "NeuralShieldingPointGoal1-v0", "NeuralShieldingPointCircle2-v0",
                                         "NeuralShieldingSauteInvertedPendulum-v4", "NeuralShieldingSautePointGoal1-v0", "NeuralShieldingSautePointCircle2-v0" ]

    need_auto_reset_wrapper = True
    need_time_limit_wrapper = False


    def __init__(self, env_id: str, device: torch.device = DEVICE_CPU, cost_budget: float = 25.0, gamma: float = 0.999, regressor: BaseEstimator = None, classifier: BaseEstimator = None, safe_agent: Any = None, **kwargs) -> None:
        '''
        '''
        self.isSaute = "Saute" in env_id
        env_id = env_id.replace("Saute", "")
        self._count = 0
        self._last_obs = None
        self._num_envs = 1
        self.regressor = regressor
        self._cost_budget = cost_budget
        self.classifier = classifier
        self.safe_agent = safe_agent
        if env_id == "NeuralShieldingInvertedPendulum-v4":
            self._inner_env = SauteInvertedPendulumEnv(cost_budget=cost_budget, gamma=gamma, isSaute=self.isSaute)
        else:
            self._inner_env = SauteSafetyGymEnv(env_id.replace("NeuralShielding", "Safety"), cost_budget=cost_budget, gamma=gamma, unsafe_reward=-100, isSaute=self.isSaute)
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
        if seed is not None:
            self.set_seed(seed)
        if isinstance(self._inner_env, SauteSafetyGymEnv):
            obs, info = self._inner_env.reset()
        else:
            obs = self._inner_env.reset_model()
            info = {}
        self._last_obs = obs
        self._count = 0
        return torch.as_tensor(obs, dtype=torch.float32, device=self._device), info

    @property
    def max_episode_steps(self) -> None:
        """The max steps per episode."""
        return 1000
    
    def render(self) -> Any:
        pass

    def close(self) -> Any:
        self._inner_env.close()

    def step(
        self,
        action: torch.Tensor,
        ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, dict]:
        self._count += 1
        action_to_perform = action.detach().numpy()
        data = np.append(self._last_obs, action_to_perform)
        data = torch.from_numpy(data).float()
        data = data.unsqueeze(0)
        data = data.to(self._device)
        with torch.no_grad():
            if self.classifier.predict(data) > 0.5:
                if (self.regressor.predict(data) / self._cost_budget) > self._inner_env._safety_state: # if the predicted cost exceeds the remaining cost budget
                    action_to_perform = self.safe_agent.predict(torch.from_numpy(self._last_obs).float()).detach().numpy() # change to action to one picked by the safe agent
        obs, reward, terminated, truncated, info = self._inner_env.step(action_to_perform)
        self._last_obs = obs

        cost = info['cost']
        truncated = self._count > 1000
        obs, reward, cost, terminated, truncated = (
            torch.as_tensor(x, dtype=torch.float32, device=self._device)
            for x in (obs, reward, cost, terminated, truncated)
        )

        if 'final_observation' in info:
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