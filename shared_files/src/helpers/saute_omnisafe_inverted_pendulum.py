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


dir_path = os.path.dirname(os.path.realpath(__file__))

DEFAULT_CAMERA_CONFIG = {
    "trackbodyid": 0,
    "distance": 2.04,
}

class SauteInvertedPendulumEnv(MujocoEnv, utils.EzPickle):
    metadata = {
        "render_modes": [
            "human",
            "rgb_array",
            "depth_array"
        ],
        "render_fps": 25,
    }

    def __init__(self, cost_budget, gamma, isSaute = True, **kwargs):
        utils.EzPickle.__init__(self, **kwargs)
        observation_space = Box(low=-np.inf, high=np.inf, shape=(5,), dtype=np.float64)
        self.obs_space_dict = {"q_pos": Box(low=-np.inf, high=np.inf, shape=(2,), dtype=np.float64), "q_vel": Box(low=-np.inf, high=np.inf, shape=(2,), dtype=np.float64), "cost_budget": Box(low=-np.inf, high=np.inf, shape=(1,), dtype=np.float64)}
        MujocoEnv.__init__(
            self,
            os.path.join(dir_path, "unlocked_inverted_pendulum.xml"),
            2,
            observation_space=observation_space,
            default_camera_config=DEFAULT_CAMERA_CONFIG,
            **kwargs,
        )
        self._cost_budget = cost_budget
        self._safety_state = 1.0
        self.gamma = gamma
        self.isSaute = isSaute

    def step(self, a):
        self.do_simulation(a, self.frame_skip)
        ob = self._get_obs()
        terminated = bool(not np.isfinite(ob).all()) 
        vertical_velocity = np.abs(ob[2])
        reward = max(1.0, vertical_velocity/4.0) # objective is to move fast horizontically
        angle = np.degrees(np.abs(ob[1])) % 360
        angle_to_bad_state = np.abs(angle - 180)

        if np.abs(ob[1]) <= 0.2: # about 11,46 degrees
            cost = 0.0
        else: # punish positions typically reached in recovery less hard
            cost = (1 - angle_to_bad_state / 180.0) ** 3
            
        if self._safety_state <= 0 and self.isSaute:
            reward = - 100.0
            terminated = True

        
        self._safety_state = (self._safety_state - cost / self._cost_budget) / self.gamma # discounted update as described in the paper
        if self.render_mode != None:
            self.render()
        return ob, reward, terminated, False, {'cost': cost}

    def reset_model(self):
        qpos = self.init_qpos + self.np_random.uniform(
            size=self.model.nq, low=-0.01, high=0.01
        )
        qvel = self.init_qvel + self.np_random.uniform(
            size=self.model.nv, low=-0.01, high=0.01
        )
        self._safety_state = 1.0
        self.set_state(qpos, qvel)
        return self._get_obs()

    def _get_obs(self):
        base_obs = np.concatenate([self.data.qpos, self.data.qvel]).ravel()
        #return base_obs
        return np.concatenate(
            [base_obs, np.array([self._safety_state], dtype=np.float32)]
        )

class SafeAgentSauteInvertedPendulumEnv():

    def __init__(self, advanced_mode=False, cost_budget=25.0, gamma=0.999, **kwargs):
        self._env = SauteInvertedPendulumEnv(cost_budget=cost_budget, gamma=gamma, **kwargs)
        self.observation_space = self._env.observation_space
        self.action_space = self._env.action_space
        self.advanced_mode = advanced_mode
        self.reset_counter = 0
    
    def step(self, a):
        ob, reward, terminated, truncated, info = self._env.step(a)
        reward = 1.0 
        if info['cost'] > 0:
            reward = -info['cost'] # for the safe agent the objective is to minimize cost. This here is reward engineering to promote safe behaviour

        return ob, reward, terminated, truncated, info
    
    def reset_model(self):
        # monte carlo sampling of initial states only in harder curriculum mode
        if self.advanced_mode and random.random() < max(0.5, self.reset_counter / 100): # 50% chance of sampling hard state in advanced mode
            self.reset_counter += 1
            qpos = self._env.init_qpos.copy()
            qpos[0] += self._env.np_random.uniform(low=-0.9, high=0.9) # actual range is from -1 to 1, but we leave a bit of margin
            qpos[1] += self._env.np_random.uniform(low=-np.pi, high=np.pi) # full range of angles

            qvel = self._env.init_qvel.copy()
            qvel[0] += self._env.np_random.uniform(low=-2.0, high=2.0) 
            qvel[1] += self._env.np_random.uniform(low=-6.0, high=6.0)
            self._env._safety_state = 1.0
            self._env.set_state(qpos, qvel)
            return self._env._get_obs()
        else:
            return self._env.reset_model()
    
    def _get_obs(self):
        return self._env._get_obs()
    
    def close(self):
        self._env.close()

class OmnisafeSauteInvertedPendulumEnv(CMDP):
    _support_envs: ClassVar[list[str]] = ["SafetyInvertedPendulum-v4", "SauteInvertedPendulum-v4", "SafeAgentBaseSauteInvertedPendulum-v4", "SafeAgentAdvancedSauteInvertedPendulum-v4"]

    need_auto_reset_wrapper = True
    need_time_limit_wrapper = False

    def __init__(self, env_id: str, device: torch.device = DEVICE_CPU, cost_budget: float = 50.0, gamma: float = 0.999, **kwargs) -> None:
        self._count = 0
        self._num_envs = 1
        if env_id == "SafetyInvertedPendulum-v4":
            self._inner_env = SauteInvertedPendulumEnv(cost_budget=cost_budget, gamma=gamma, isSaute=False)
        elif env_id == "SauteInvertedPendulum-v4":
            self._inner_env = SauteInvertedPendulumEnv(cost_budget=cost_budget, gamma=gamma)
        elif env_id == "SafeAgentAdvancedSauteInvertedPendulum-v4":
            self._inner_env = SafeAgentSauteInvertedPendulumEnv(advanced_mode=True, cost_budget=cost_budget, gamma=gamma)
        else:
            self._inner_env = SafeAgentSauteInvertedPendulumEnv(advanced_mode=False, cost_budget=cost_budget, gamma=gamma)
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
        obs = self._inner_env.reset_model()
        self._count = 0
        return torch.as_tensor(obs, dtype=torch.float32, device=self._device), {}

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
        obs, reward, terminated, truncated, info = self._inner_env.step(action.detach().cpu().numpy())
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
