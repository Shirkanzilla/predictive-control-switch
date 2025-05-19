import torch
import torch.nn as nn
import torch.nn.functional as F
import safety_gymnasium
import numpy as np
from pyvirtualdisplay import Display
display = Display(visible=0, size=(1400, 900))
display.start()

class CustomPolicy(nn.Module):
    def __init__(self, obs_dim, act_dim):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(obs_dim, 64),
            nn.ReLU(),
            nn.Linear(64, 64),
            nn.ReLU(),
            nn.Linear(64, act_dim),
            nn.Tanh()  # Assuming action space is in [-1, 1]
        )

    def forward(self, obs):
        return self.net(obs)

env = safety_gymnasium.make('SafetyPointGoal1-v0')
obs, _ = env.reset()

obs_dim = env.observation_space.shape[0]
act_dim = env.action_space.shape[0]

def random_model():
    model = CustomPolicy(obs_dim, act_dim)
    model.apply(lambda m: nn.init.kaiming_uniform_(m.weight) if isinstance(m, nn.Linear) else None)
    return model

model = random_model()

# Random interaction loop
done = False
while not done:
    obs_tensor = torch.tensor(obs, dtype=torch.float32)
    with torch.no_grad():
        action = model(obs_tensor).numpy()

    # Clip to valid action range
    action = np.clip(action, env.action_space.low, env.action_space.high)

    obs, reward, terminated, truncated, info = env.step(action)
    done = terminated or truncated
    print(done)

    #env.render()  # Optional