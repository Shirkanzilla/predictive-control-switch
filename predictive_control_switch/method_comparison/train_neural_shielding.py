import omnisafe
import torch
from safety_gymnasium import make
from helpers.register_envs import register_envs
from helpers.load_model import load_guide
import sys
import pickle
import json
register_envs()

# Train
if len(sys.argv) < 4:
    print("Usage: python train_neural_shielding.py [env_id] [algorithm] [safe_agent_dir_path] [pt-file name] [epochs] [device] [threads] ")
    sys.exit(1)

env_id = sys.argv[1]
algorithm = sys.argv[2]
safe_agent_dir_path = sys.argv[3]
pt_file_name = sys.argv[4]
epochs = sys.argv[5]
device = sys.argv[6] if len(sys.argv) > 6 else "cpu"
threads = sys.argv[7] if len(sys.argv) > 7 else 20

with open(f"{safe_agent_dir_path}/regressor.pkl", 'rb') as f:
    regressor = pickle.load(f)

with open(f"{safe_agent_dir_path}/classifier.pkl", 'rb') as f:
    classifier = pickle.load(f)

safe_agent = load_guide(safe_agent_dir_path, pt_file_name)[1]

custom_cfgs = {
    'train_cfgs': {
        'torch_threads': int(threads), 
        'total_steps': 20000 * int(epochs),
        'device' : device,
    },
}
agent = omnisafe.Agent(algorithm, env_id=env_id, custom_cfgs=custom_cfgs)
# Workaround to inject parameters as we could not get the official method to work (via env_kwargs)
# Might now be super robust
agent.agent._env._env._env._env._env._env.regressor = regressor
agent.agent._env._env._env._env._env._env.classifier = classifier
agent.agent._env._env._env._env._env._env.safe_agent = safe_agent

agent.learn()
agent.plot(smooth=1)