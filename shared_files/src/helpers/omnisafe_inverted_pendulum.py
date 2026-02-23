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

class SafetyInvertedPendulumEnv(MujocoEnv, utils.EzPickle):
    metadata = {
        "render_modes": [
            "human",
            "rgb_array",
            "depth_array"
        ],
        "render_fps": 25,
    }


    def __init__(self, **kwargs):
        utils.EzPickle.__init__(self, **kwargs)
        observation_space = Box(low=-np.inf, high=np.inf, shape=(4,), dtype=np.float64)
        MujocoEnv.__init__(
            self,
            os.path.join(dir_path, "unlocked_inverted_pendulum.xml"),
            2,
            observation_space=observation_space,
            default_camera_config=DEFAULT_CAMERA_CONFIG,
            **kwargs,
        )

    def step(self, a):
        self.do_simulation(a, self.frame_skip)
        ob = self._get_obs()
        terminated = bool(not np.isfinite(ob).all()) # or (np.abs(ob[1]) > 1))
        reward = 0.0
        angle = np.degrees(np.abs(ob[1])) % 360
        if np.abs(ob[1]) <= 0.2: # about 11,46 degrees
            cost = 0.0
            reward = 1.0
        elif np.abs(angle - 180) > 30: # punish positions typically reached in recovery less hard
            cost = 0.1
        else:
            cost = 1.0
        #if np.abs(ob[1]) > 0.2:
        #   cost = min(np.abs(ob[1]), 1)
        #else:
        #    reward = 1.0
        if self.render_mode != None:
            self.render()
        return ob, reward, cost, terminated, False, {}

    def reset_model(self):
        qpos = self.init_qpos + self.np_random.uniform(
            size=self.model.nq, low=-0.01, high=0.01
        )
        qvel = self.init_qvel + self.np_random.uniform(
            size=self.model.nv, low=-0.01, high=0.01
        )
        self.set_state(qpos, qvel)
        return self._get_obs()

    def _get_obs(self):
        base_obs = np.concatenate([self.data.qpos, self.data.qvel]).ravel()
        return base_obs
    
class OmnisafeInvertedPendulumEnv(CMDP):
    _support_envs: ClassVar[list[str]] = ["SafetyInvertedPendulum-v4"]

    need_auto_reset_wrapper = True
    need_time_limit_wrapper = False

    def __init__(self, env_id: str, device: torch.device = DEVICE_CPU, **kwargs) -> None:
        self._count = 0
        self._num_envs = 1
        self._inner_env = SafetyInvertedPendulumEnv()
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
        obs, reward, cost, terminated, truncated, info = self._inner_env.step(action.detach().cpu().numpy())
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
