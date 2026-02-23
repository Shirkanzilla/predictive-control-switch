from helpers.register_envs import register_envs
from omnisafe.common.offline.data_collector import OfflineDataCollector
import numpy as np
import pandas as pd

register_envs()

env_name = 'SauteSafetyInvertedPendulum-v4'
size = 100_000
agents = [
    ('./runs/PPO-{SauteSafetyInvertedPendulum-v4}/100epochs_gradual_cf/', 'epoch-100.pt', 100_000),
]
save_dir = './runs/PPO-{SauteSafetyInvertedPendulum-v4}/100epochs_gradual_cf/'

if __name__ == '__main__':
    col = OfflineDataCollector(size, env_name)
    for agent, model_name, num in agents:
        col.register_agent(agent, model_name, num)
    col.collect(save_dir)

data = np.load(save_dir + "/" + env_name + "_data.npz")

df = pd.DataFrame({'reward': data['reward'].squeeze(), 'cost': data['cost'].squeeze()})
print(df.describe())