import sys

from helpers.register_envs import register_envs
from omnisafe.common.offline.data_collector import OfflineDataCollector
from omnisafe.adapter import SauteAdapter
from omnisafe.envs.wrapper import ActionScale
from omnisafe.utils.config import Config, get_default_kwargs_yaml
import torch
import numpy as np
import pandas as pd

class SauteOfflineDataCollector(OfflineDataCollector):
    def __init__(self, size: int, env_name: str) -> None:
        super().__init__(size, env_name)

        self._env = SauteAdapter(env_name, 1, 42, get_default_kwargs_yaml('PPOSaute', env_name, 'on-policy'))
        self._obs_space = self._env.observation_space
        print("Observation space:", self._obs_space)
        self._action_space = self._env.action_space

        self._env = ActionScale(self._env, device=torch.device('cpu'), high=1.0, low=-1.0)  

        self._obs = np.zeros((size, *self._obs_space.shape), dtype=np.float32)
        self._next_obs = np.zeros((size, *self._obs_space.shape), dtype=np.float32)

register_envs()

if len(sys.argv) < 4:
    print("Usage: python evaluate_policy.py [env_id] [algorithm] [seed/subdir name] [epoch]")
    sys.exit(1)

env_id = sys.argv[1]
algorithm = sys.argv[2]
seed = sys.argv[3]
epoch = sys.argv[4]

size = 100_000
save_dir = f'./runs/{algorithm}-{{{env_id}}}/{seed}/'
agents = [
    (save_dir, f'epoch-{epoch}.pt', 100_000),
]

env_id = env_id.replace("SafeAgentBase", "").replace("SafeAgentAdvanced", "") # We want to evaluate the policy on the original environment, not the monte carlo sampling wrapper
env_id = env_id.replace("SafeAgent", "")

if __name__ == '__main__':
    if algorithm == 'PPOSaute':
        print("Using SauteOfflineDataCollector")
        col = SauteOfflineDataCollector(size, env_id)
    else:
        print(env_id)
        col = OfflineDataCollector(size, env_id)
    for agent, model_name, num in agents:
        col.register_agent(agent, model_name, num)
    col.collect(save_dir)

data = np.load(save_dir + "/" + env_id + "_data.npz")

df = pd.DataFrame({'reward': data['reward'].squeeze(), 'cost': data['cost'].squeeze()})
df['cost_budget'] = data['obs'][:, -1].squeeze()
print(df.describe())

df_budget_exceeded= df.where(df['reward'] == -100)

df_trajectory_starts = df.where(df['cost_budget'] == df['reward'])

print("Percentage of trajectories that exceeded the cost budget:", df_budget_exceeded.count()[0] / df_trajectory_starts.count()[0])
print("Percentage of trajectories in cost budget:", 1 - df_budget_exceeded.count()[0] / df_trajectory_starts.count()[0])