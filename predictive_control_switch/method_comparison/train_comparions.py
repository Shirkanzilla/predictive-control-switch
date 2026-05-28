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
    print("Usage: python train_comparisons.py [env_id] [epochs] [device] [threads] ")
    sys.exit(1)

env_id = sys.argv[1]
epochs = sys.argv[2]
device = sys.argv[3] if len(sys.argv) > 6 else "cpu"
threads = sys.argv[4] if len(sys.argv) > 7 else 20


custom_cfgs = {
    'train_cfgs': {
        'torch_threads': int(threads), 
        'total_steps': 20000 * int(epochs),
        'device' : device,
    },
}
eg = ExperimentGrid(exp_name="NeuralShielding")
for algorithm in ['PPOLag', 'PPOSaute']:
    if 'Saute' in algorithm:
        agent = omnisafe.Agent(algorithm.replace("Saute", ""), env_id=env_id.replace("Safety", "Saute"), custom_cfgs=custom_cfgs)
    else:
        agent = omnisafe.Agent(algorithm, env_id=env_id, custom_cfgs=custom_cfgs)

    agent.learn()
    agent.plot(smooth=1)
