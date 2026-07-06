from __future__ import annotations
import omnisafe
import torch
import numpy as np
import os
import pickle
from helpers.register_envs import register_envs
from helpers.load_model import load_guide
import sys

register_envs()

if len(sys.argv) < 4:
    print("Usage: python train_all_variants_parallel.py [base_env_id] [safe_agent_dir_path] [pt-file name] [epochs]")
    sys.exit(1)

base_env_id = sys.argv[1] 
safe_agent_dir_path = sys.argv[2]
pt_file_name = sys.argv[3]
epochs = int(sys.argv[4])

with open(f"{safe_agent_dir_path}/models_dict.pkl", 'rb') as f:
    models_dict = pickle.load(f)

safe_agent = load_guide(safe_agent_dir_path, pt_file_name)[1]

env_configs = [
    {
        "name": "NeuralShieldingInvertedPendulum",
        "env_id": f"NeuralShielding{base_env_id}",
        "algo": "PPOLag",
        "use_custom_params": True,
    },
    {
        "name": "SauteInvertedPendulum",
        "env_id": f"Safety{base_env_id}",
        "algo": "PPOSaute",
        "use_custom_params": False,
    },
    {
        "name": "SafetyInvertedPendulum",
        "env_id": f"Safety{base_env_id}",
        "algo": "PPOLag",
        "use_custom_params": False,
    },
]


def train_agent(env_config, seed, models_dict, safe_agent, epochs):
    env_name = env_config["name"]
    env_id = env_config["env_id"]
    algo = env_config["algo"]
    use_custom_params = env_config["use_custom_params"]

    torch.manual_seed(seed)
    np.random.seed(seed)

    # Create log dir
    log_dir = f"./results/{env_name}/seed_{seed}"
    os.makedirs(log_dir, exist_ok=True)

    custom_cfgs = {
        'train_cfgs': {
            'total_steps': 20000 * epochs,
            'device': 'cpu',
        },
        'logger_cfgs':{
            'log_dir': log_dir,
        }
    }

    if algo == "PPOSaute":
        custom_cfgs.update({"algo_cfgs": {"safety_budget": 50.0}})

    agent = omnisafe.Agent(algo, env_id=env_id, custom_cfgs=custom_cfgs)

    if use_custom_params:
        print(f"Injecting custom parameters into {env_name} (seed={seed})")
        env = agent.agent._env
        while hasattr(env, '_env'):
            env = env._env
        env.regressor = models_dict['regressor']
        env.classifier = models_dict['classifier']
        env.scaler = models_dict['scaler_X']
        env.safe_agent = safe_agent

    print(f"Training {algo} on {env_id} (seed={seed})")
    reward, cost, ep_len = agent.learn()

    return {
        "env": env_name,
        "seed": seed,
        "reward": reward,
        "cost": cost,
        "ep_len": ep_len,
    }

if __name__ == "__main__":
    # Train each environment with 5 seeds sequentially
    seeds = [0, 1] #, 2, 3, 4]
    for env_config in env_configs:
        env_name = env_config["name"]
        print(f"\n--- Training {env_name} ---")

        for seed in seeds:
            try:
                result = train_agent(
                    env_config,
                    seed,
                    models_dict,
                    safe_agent,
                    epochs,
                )
                print(
                    f"  Seed {seed}: reward={result['reward']:.2f}, "
                    f"cost={result['cost']:.2f}, ep_len={result['ep_len']:.2f}"
                )
            except Exception as e:
                print(f"  ERROR for seed {seed}: {e}")

    print("\nTraining complete! Results saved in ./results/")