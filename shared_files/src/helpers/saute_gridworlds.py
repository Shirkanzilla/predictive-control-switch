from __future__ import annotations

import random
from typing import Any, ClassVar

import torch
from gymnasium import spaces

from omnisafe.envs.core import CMDP
from omnisafe.typing import DEVICE_CPU, Box

import numpy as np

from gymnasium.envs.toy_text.frozen_lake import FrozenLakeEnv
from gymnasium.envs.toy_text.utils import categorical_sample
from gymnasium.spaces import Box

import omnisafe

class CustomFrozenLake(FrozenLakeEnv):
    metadata = {
        "render_modes": ["human", "ansi", "rgb_array"],
        "render_fps": 4,
    }
    def __init__(self, cost_budget, gamma, isSaute = True, **kwargs):
        super().__init__(map_name="8x8", is_slippery=False, render_mode="rgb_array")
        self.obs_space_dict = {}
        for i in range(64):
            self.obs_space_dict[f"state_{i}"] =  Box(low=0.0, high=1.0, shape=(1,), dtype=np.float32)
        if isSaute:
            self.obs_space_dict["cost_budget"] =  Box(low=0.0, high=np.inf, shape=(1,), dtype=np.float32)
            # one hot encoding of the state space + the cost budget
            low = np.zeros(65, dtype=np.float32)
            high = np.ones(65, dtype=np.float32)
            high[-1] = np.inf
            self.observation_space = spaces.Box(low=low, high=high, dtype=np.float32)
        else:
            self.observation_space = spaces.Box(
                low=0.0,
                high=1.0,
                shape=(64,),
                dtype=np.float32
            ) 
        self.action_space = spaces.Box(
                low=0.0,
                high=1.0,
                shape=(4,),
                dtype=np.float32
        )
        self._cost_budget = cost_budget
        self._safety_state = 1.0
        self.gamma = gamma
        self.isSaute = isSaute

    def step(self, a):
        if np.nan in a:
            print(a)
        #a = np.nan_to_num(a, nan=0.0)
        a_noised = a + np.random.uniform(0, 1e-8, size=a.shape) # break ties if values for different directions are identical
        state, reward, terminated, truncated, info = super().step(np.argmax(a_noised)) # pick the highest valued action
        ob = self._get_obs(state)
        assert np.all(np.isfinite(ob))
        cost = 1.0 if terminated and reward == 0 else 0.0
        if terminated:
            terminated = False
            
        self._safety_state = (self._safety_state - cost / self._cost_budget) / self.gamma # discounted update as described in the paper

        if self._safety_state <= 0 and self.isSaute:
            reward = - 100.0
            terminated = True
        
        if self.render_mode != None:
            self.render()
        
        info['cost'] = cost
        return ob, reward, terminated, truncated, info
    
    def reset(
        self,
        *,
        seed = None,
        options = None,
    ):
        s, info = super().reset(seed=seed)
        self._safety_state = 1.0
        return self._get_obs(s), {'cost': 0, 'prob': 1}

    def _get_obs(self, state):
        assert np.isfinite(self._safety_state), self._safety_state
        assert isinstance(state, int), state
        base_obs = np.zeros(64)
        base_obs[state] = 1.0
        assert np.count_nonzero(base_obs) == 1, base_obs
        if self.isSaute:
            return np.concatenate(
                [base_obs, np.array([self._safety_state], dtype=np.float32)]
            )
        else:
            return base_obs

class SafeAgentWrapper():
    metadata = {
        "render_modes": ["human", "ansi", "rgb_array"],
        "render_fps": 4,
    }
    def __init__(self, cost_budget=25.0, gamma=0.999, **kwargs):
        self._env = CustomFrozenLake(cost_budget=cost_budget, gamma=gamma, **kwargs)
        self.observation_space = self._env.observation_space
        self.action_space = self._env.action_space
    
    def step(self, a):
        ob, reward, terminated, truncated, info = self._env.step(a)
        if reward >= 0:
            reward = 0.01 - info["cost"] # small reward for safe steps
        return ob, reward, terminated, truncated, info
    
    def reset(self):
        # randomly sample a starting state over all possible states
        self._env.reset()
        state_distribution = np.ones(self._env.desc.shape).ravel()
        state_distribution /= state_distribution.sum()
        self._env.s = categorical_sample(state_distribution, self._env.np_random)
        # go one random step from it to get correct state and cost info
        a = np.zeros(4)
        a[np.random.randint(0,4)] = 1
        # (maybe later) randomize safety state to cover the full range the agent might inherit
        # elf._env._safety_state = np.random.uniform(0.1, 1.0)
        ob, _, _, _, info = self._env.step(a)
        return ob, info
    
    def render(self):
        return self._env.render()
    
    def _get_obs(self, s):
        return self._env._get_obs(s)
    
    def close(self):
        self._env.close()

class OmnisafeGridWorldsEnv(CMDP):
    _support_envs: ClassVar[list[str]] = ["SafetyFrozenLake", "SauteFrozenLake", "SafeAgentSauteFrozenLake"]

    need_auto_reset_wrapper = True
    need_time_limit_wrapper = False

    def __init__(self, env_id: str, device: torch.device = DEVICE_CPU, cost_budget: float = 3.0, gamma: float = 0.99, **kwargs) -> None:
        self._count = 0
        self._num_envs = 1
        if env_id == "SauteFrozenLake":
            self._inner_env = CustomFrozenLake(cost_budget=cost_budget, gamma=gamma)
        elif env_id == "SafeAgentSauteFrozenLake":
            self._inner_env = SafeAgentWrapper(cost_budget=cost_budget, gamma=gamma)
        else:
            self._inner_env = CustomFrozenLake(cost_budget=cost_budget, gamma=gamma, isSaute=False)
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
        obs, info = self._inner_env.reset()
        self._count = 0
        return torch.as_tensor(obs, dtype=torch.float32, device=self._device), info

    @property
    def max_episode_steps(self) -> None:
        """The max steps per episode."""
        return 100
    
    def render(self) -> Any:
        return self._inner_env.render()

    def close(self) -> Any:
        self._inner_env.close()

    def step(
        self,
        action: torch.Tensor,
        ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, dict]:
        self._count += 1
        obs, reward, terminated, truncated, info = self._inner_env.step(action.detach().cpu().numpy())
        cost = info['cost']
        truncated = self._count > 100
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