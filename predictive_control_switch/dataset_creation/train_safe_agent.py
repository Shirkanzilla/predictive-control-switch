import omnisafe
from helpers.register_envs import register_envs
import sys

register_envs()

# Train
if len(sys.argv) < 4:
    print("Usage: python train_safe_agent.py [env_id] [algorithm] [epochs]")
    sys.exit(1)

env_id = sys.argv[1]
algorithm = sys.argv[2]
epochs = sys.argv[3]

# Train for 50 epochs (default steps is 20.000, so i lowered the total steps from 10.000.000 to 1.000.000) https://github.com/PKU-Alignment/omnisafe/blob/main/omnisafe/configs/on-policy/PPOLag.yaml for reference
custom_cfgs = {
    'train_cfgs': {
        'torch_threads': 20, 
        'total_steps': 20000 * int(epochs),
        'device' : "cpu",
    }
}

agent = omnisafe.Agent(algorithm, env_id=env_id, custom_cfgs=custom_cfgs)
agent.learn()

agent.plot(smooth=1)
