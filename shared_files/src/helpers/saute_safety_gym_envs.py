from __future__ import annotations

import random
from typing import Any, ClassVar

import torch
from gymnasium import spaces
import safety_gymnasium
import mujoco

from omnisafe.envs.core import CMDP
from omnisafe.typing import DEVICE_CPU

import numpy as np

from dataclasses import dataclass

@dataclass
class RangeSpec:
    low: float
    high: float

class SauteSafetyGymEnv():
    metadata = {
        "render_modes": [
            "human",
            "rgb_array",
            "depth_array"
        ],
        "render_fps": 25,
    }

    def __init__(self, env_id, cost_budget=25.0, gamma=0.999, unsafe_reward=-100.0, isSaute = True, **kwargs):
        self._env = safety_gymnasium.make(env_id, render_mode=kwargs.get("render_mode", "rgb_array"))
        self._initial_cost_budget = cost_budget
        self._unsafe_reward = unsafe_reward
        self._cost_budget = 1.0
        self._gamma = gamma
        self.isSaute = isSaute

        obs_space = self._env.observation_space
        low = np.concatenate([obs_space.low, [0.0]])
        high = np.concatenate([obs_space.high, [np.inf]])

        self.observation_space = spaces.Box(low=low, high=high, dtype=np.float32)
        self.action_space = self._env.action_space

    def step(self, a):
        obs, reward, cost, terminated, truncated, info = self._env.step(a)

        if self._cost_budget <= 0 and self.isSaute:
            terminated = True
            reward = self._unsafe_reward

        self._cost_budget = (self._cost_budget - cost / self._initial_cost_budget) / self._gamma
        
        obs = np.concatenate([obs, np.array([self._cost_budget], dtype=np.float32)])

        return obs, reward, cost, terminated, truncated, info

    def reset(self, seed=None, options=None):
        obs, info =self._env.reset(seed=seed, options=options)
        self._cost_budget = 1.0
        obs = np.concatenate([obs, np.array([self._cost_budget], dtype=np.float32)])

        return obs, info
    
    def close(self):
        self._env.close()
    
    def render(self):
        return self._env.render()

class SafeAgentSauteSafetyGymEnv(SauteSafetyGymEnv):
    def __init__(self, env_id, cost_budget=25.0, gamma=0.999, unsafe_reward=-100.0, **kwargs):
        super().__init__(env_id, cost_budget, gamma, unsafe_reward, **kwargs)
        self.qpos_ranges={
            0: RangeSpec(-1.5, 1.5),     # x
            1: RangeSpec(-1.5, 1.5),     # y
            2: RangeSpec(-np.pi, np.pi), # yaw
        }
        self.qvel_ranges={
            0: RangeSpec(-2.0, 2.0),
            1: RangeSpec(-2.0, 2.0),
        }
        unwrapped = self._env.unwrapped
        self.model = unwrapped.task.agent.engine.model
        self.data = unwrapped.task.agent.engine.data

    def step(self, a):
        obs, reward, cost, terminated, truncated, info = super().step(a)
        reward = - cost
        if obs[-1] <= 0:
            reward = min(reward, self._unsafe_reward)
        return obs, reward, cost, terminated, truncated, info
    
    def _sample_uniform(self, spec: RangeSpec, rng: np.random.Generator) -> float:
        return float(rng.uniform(spec.low, spec.high))

    def _set_state(self, qpos, qvel) -> None:
        assert qpos.shape == (self.model.nq,) and qvel.shape == (self.model.nv,)

        self.data.qpos[:] = np.copy(qpos)
        self.data.qvel[:] = np.copy(qvel)
        if self.model.na == 0:
            self.data.act[:] = None
        mujoco.mj_forward(self.model, self.data)

    
    def reset(self, *, seed=None, options=None):
        obs, info = super().reset(seed=seed, options=options)
        rng = np.random.default_rng(seed)

        model, data = self.model, self.data

        qpos = data.qpos.copy()
        qvel = data.qvel.copy()

        for i, spec in self.qpos_ranges.items():
            qpos[i] = self._sample_uniform(spec, rng)

        for i, spec in self.qvel_ranges.items():
            qvel[i] = self._sample_uniform(spec, rng)

        # MuJoCo state update
        self._set_state(qpos, qvel)

        obs = self._env.unwrapped.task.obs()
        self._cost_budget = 1.0
        obs = np.concatenate([obs, np.array([self._cost_budget], dtype=np.float32)])        
        return obs, info



class OmnisafeSauteSafetyGymEnvs(CMDP):
    _support_envs: ClassVar[list[str]] = ["SautePointGoal1-v0", "SautePointCircle2-v0", "SautePointFormulaOne1-v0", "SafeAgentSautePointGoal1-v0", "SafeAgentSautePointCircle2-v0", "SafeAgentSautePointFormulaOne1-v0"]

    need_auto_reset_wrapper = True
    need_time_limit_wrapper = False

    def __init__(self, env_id: str, num_envs: int = 1, device: torch.device = DEVICE_CPU, **kwargs) -> None:
        print("Using Saute OmniSafe environment")
        self._count = 0
        self._num_envs = 1 # not supported
        env_id = env_id.replace("Saute", "Safety")
        if "SafeAgent" in env_id:
            self._inner_env = SafeAgentSauteSafetyGymEnv(env_id=env_id.replace("SafeAgent", ""), cost_budget=25.0, gamma=0.999, unsafe_reward=-100.0, **kwargs)
        else:
            self._inner_env = SauteSafetyGymEnv(env_id=env_id, cost_budget=25.0, gamma=0.999, unsafe_reward=-100.0, **kwargs)
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
        self._count = 0
        return torch.as_tensor(obs, dtype=torch.float32, device=self._device), info

    @property
    def max_episode_steps(self) -> None:
        """The max steps per episode."""
        return self._inner_env.spec.max_episode_steps
    
    def render(self) -> Any:
        return self._inner_env.render()

    def close(self) -> Any:
        self._inner_env.close()

    def step(
        self,
        action: torch.Tensor,
        ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, dict]:
        obs, reward, cost, terminated, truncated, info = self._inner_env.step(
            action.detach().numpy(), # this is necessary for evaluate_policy.py but slows things down as well I think
        )
        #terminated = reward < 0.0
        #truncated = self._count > self.max_episode_steps
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
 #       print(f"Step: {self._count} with reward {reward.item()} and cost {cost.item()}, terminated: {terminated}, truncated: {truncated}.")
        self._count += 1

        return obs, reward, cost, terminated, truncated, info
