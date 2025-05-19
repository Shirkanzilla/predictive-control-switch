import omnisafe
import torch
print("CUDA available:", torch.cuda.is_available())
print("CUDA device count:", torch.cuda.device_count())
print("Current device:", torch.cuda.current_device())
print("Device name:", torch.cuda.get_device_name(torch.cuda.current_device()))

env_id = 'SafetyPointGoal1-v0'
# Train for 100 epochs (default steps is 20.000, so i lowered the total steps from 10.000.000 to 1.000.000) https://github.com/PKU-Alignment/omnisafe/blob/main/omnisafe/configs/on-policy/PPOLag.yaml for reference
custom_cfgs = {
    'train_cfgs': {
        'total_steps': 1000000,
        'device' : "cpu",
    },
}

agent = omnisafe.Agent("PPOLag", env_id=env_id, custom_cfgs=custom_cfgs)
agent.learn()

agent.plot(smooth=1)