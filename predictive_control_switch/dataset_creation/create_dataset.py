from time import time

import numpy as np
from omnisafe.models.actor import GaussianLearningActor
import safety_gymnasium
from gymnasium import spaces
import gymnasium
import torch
from load_model import load_guide
from tqdm import tqdm
import pandas as pd
import sys
from multiprocessing import Pool
from functools import partial

from helpers.register_envs import register_envs
from helpers.saute_omnisafe_inverted_pendulum import SauteInvertedPendulumEnv
from helpers.saute_safety_gym_envs import SauteSafetyGymEnv

def create_random_agent(env, hidden_layers=[255,255,255,255], activation='relu', weight_initialization_mode='orthogonal'):
    obs_space = env.observation_space
    act_space = env.action_space
    return GaussianLearningActor(obs_space, act_space, hidden_layers, activation=activation, weight_initialization_mode=weight_initialization_mode)

def run_trajectory(env, agent, safe_agent, min_rand_steps=100, max_rand_steps=400, cost_window=200, deterministic=True):
    observation, info = env.reset()
    episode_over = False
    sampled_cost = 0
    sampling_step = 0
    agent_instance_for_pos = None
    try:
        if not isinstance(env.unwrapped, SauteInvertedPendulumEnv):
            agent_instance_for_pos = env.unwrapped.__getattribute__("task").agent
    except:
        pass
    data = []
    label = None
    # get a random number for the amount of steps the random agent should take before a sample is created
    num_steps = np.random.randint(min_rand_steps, max_rand_steps)
    # gather data
    for i in range(num_steps):
        if agent_instance_for_pos is not None:
            # Discard trajectory if the agent moves out of the checkered 7x7 space, coordinates were tested manually
            if abs(agent_instance_for_pos.pos[0]) >= 3.5 or abs(agent_instance_for_pos.pos[1]) >= 3.5: 
                break
        obs_tensor = torch.from_numpy(observation).float()
        action = agent.predict(obs_tensor, deterministic=deterministic).detach().numpy()
        if i == num_steps - 1:
            data = np.append(observation, action) 
        observation, reward, cost, terminated, truncated, info = env.step(action)
        episode_over = terminated or truncated
        if episode_over:
            break
    if not episode_over:
        # sample with the pre trained agent
        for i in range(cost_window):
            obs_tensor = torch.from_numpy(observation).float()
            action = safe_agent.predict(obs_tensor, deterministic=deterministic).detach().numpy()
            observation, reward, cost, terminated, truncated, info = env.step(action)
            episode_over = terminated or truncated
            sampled_cost += cost
            sampling_step += 1
            if episode_over or i == cost_window - 1:
                label = sampled_cost
                break
    env.close()
    return data, label

def init_worker(env_id):
    global env
    if env_id == "SauteInvertedPendulum-v4":
        env = safety_gymnasium.wrappers.Gymnasium2SafetyGymnasium(gymnasium.make(env_id, cost_budget=25.0, gamma=0.99))
    elif "Saute" in env_id:
        env = SauteSafetyGymEnv(env_id.replace("Saute", "Safety"), render_mode=None)
    else:
        env = safety_gymnasium.make(env_id, render_mode=None)

def generate_sample(safe_agent, min_rand_steps=100, max_rand_steps=400, cost_window=200, deterministic=True, pool_iterator=None):
    global env
    data = []
    label = None
    while len(data) == 0 or label == None:
        data, label = run_trajectory(env, create_random_agent(env), safe_agent, min_rand_steps, max_rand_steps, cost_window, deterministic)
    return data, label


def generate_dataset(env_id, safe_agent, amount=1000, min_rand_steps=100, max_rand_steps=400, cost_window=200, deterministic=True, n_processes=4):
    start_time = time()
    data = []
    labels = []
    generate_sample_partial = partial(generate_sample, safe_agent, min_rand_steps, max_rand_steps, cost_window, deterministic)
    with Pool(processes=n_processes, initializer=init_worker, initargs=(env_id,)) as pool:
        results =pool.map(generate_sample_partial, range(amount))
        for data_point, label in results:
            if len(data_point) > 0 and label is not None:
                data.append(data_point)
                labels.append(label)
    end_time = time()
    print(f"Dataset generation took {(end_time - start_time)/amount:.2f} seconds per sample on average.")
    return data, labels

if __name__ == "__main__":
    np.random.seed(42)
    torch.manual_seed(42)

    register_envs()

    if len(sys.argv) < 4:
        print("Usage: python create_dataset.py <env_id> <amount> <safe_agent_dir> <pt_file_name> [min_rand_steps] [max_rand_steps] [cost_window] [deterministic] [n_processes]")
        sys.exit(1)

    env_id = sys.argv[1]
    amount = int(sys.argv[2])
    safe_agent_dir = sys.argv[3]
    pt_file_name = sys.argv[4]
    min_rand_steps = int(sys.argv[5]) if len(sys.argv) > 5 else 100
    max_rand_steps = int(sys.argv[6]) if len(sys.argv) > 6 else 400
    cost_window = int(sys.argv[7]) if len(sys.argv) > 7 else 200
    deterministic = bool(int(sys.argv[8])) if len(sys.argv) > 8 else True
    n_processes = int(sys.argv[9]) if len(sys.argv) > 9 else 1

    safe_agent = load_guide(safe_agent_dir, pt_file_name)[1]

    data, labels = generate_dataset(env_id, safe_agent, amount=amount, min_rand_steps=min_rand_steps, max_rand_steps=max_rand_steps, cost_window=cost_window, deterministic=deterministic, n_processes=n_processes)

    # instantiate the env here again to get obs and action space for column names
    if env_id == "SauteInvertedPendulum-v4":
        env = safety_gymnasium.wrappers.Gymnasium2SafetyGymnasium(gymnasium.make(env_id, cost_budget=25.0, gamma=0.99))
        obs_space_dict = env.obs_space_dict 
    elif "Saute" in env_id:
        env = SauteSafetyGymEnv(env_id.replace("Saute", "Safety"), render_mode=None)
        obs_space_dict = env._env.obs_space_dict
        obs_space_dict['safety_state'] = spaces.Box(-np.inf, np.inf)
    else:
        env = safety_gymnasium.make(env_id, render_mode=None)
        obs_space_dict = env.obs_space_dict 

    obs_columns = []
    for key in obs_space_dict.keys():
        for i in range(obs_space_dict[key].shape[0]):
            obs_columns.append(f'{key}_{i}')
    for i in range(env.action_space.shape[0]):
        obs_columns.append(f'action_{i}')

    env.close()

    df = pd.DataFrame(data=data, columns=obs_columns)
    df['exp_cost'] = labels
    df.head()

    df.to_pickle(f"{safe_agent_dir}/dataset.pkl")