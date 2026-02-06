import omnisafe
from omnisafe.envs.core import env_register, env_unregister
from omnisafe_inverted_pendulum import OmnisafeInvertedPendulumEnv
import safety_gymnasium
import torch
print("CUDA available:", torch.cuda.is_available())
print("CUDA device count:", torch.cuda.device_count())
print("Current device:", torch.cuda.current_device())
print("Device name:", torch.cuda.get_device_name(torch.cuda.current_device()))

safety_gymnasium.register(id="SafetyInvertedPendulum-v4",
    entry_point="omnisafe_inverted_pendulum:SafetyInvertedPendulumEnv",
    max_episode_steps=1000,
    reward_threshold=950.0,
)

@env_register
@env_unregister
class OmnisafeInvertedPendulum(OmnisafeInvertedPendulumEnv):
    pass

env = OmnisafeInvertedPendulum("SafetyInvertedPendulum-v4")

env_id = 'SafetyInvertedPendulum-v4'
# Train for 50 epochs (default steps is 20.000, so i lowered the total steps from 10.000.000 to 1.000.000) https://github.com/PKU-Alignment/omnisafe/blob/main/omnisafe/configs/on-policy/PPOLag.yaml for reference
custom_cfgs = {
    'train_cfgs': {
        'total_steps': 1000000,
        'device' : "cpu",
    },
    'lagrange_cfgs':{
        'cost_limit': 15.00 # default is 25.00
    }
}

agent = omnisafe.Agent("PPOLag", env_id=env_id, custom_cfgs=custom_cfgs)
agent.learn()

agent.plot(smooth=1)
