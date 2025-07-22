import os
# bugfix for my setup
os.environ["DISPLAY"] = ":0"
from pynput.keyboard import Listener, KeyCode
import safety_gymnasium
from time import sleep
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

class ExpectedCostRegressor(nn.Module):
    def __init__(self):
        super(ExpectedCostRegressor, self).__init__()
        self.fc = nn.Sequential(
            nn.Linear(57, 256),
            nn.ReLU(),
            nn.Linear(256, 256),
            nn.ReLU(),
            nn.Linear(256,1),
        )
    def forward(self, x):
        return self.fc(x)

class ExpectedCostClassifier(nn.Module):
    def __init__(self):
        super(ExpectedCostClassifier, self).__init__()
        self.fc = nn.Sequential(
            nn.Linear(57, 128),
            nn.ReLU(),
            nn.Linear(128, 128),
            nn.ReLU(),
            nn.Linear(128, 128),
            nn.ReLU(),
            nn.Linear(128, 128),
            nn.ReLU(),
            nn.Linear(128, 128),
            nn.ReLU(),
            nn.Linear(128,1),
            nn.Sigmoid()
        )
    def forward(self, x):
        return self.fc(x)
    
class InputAction:
    def __init__(self):
        self.keys = {'i': False, 'j': False, 'k': False, 'l': False}
        self.listener = Listener(on_press=self.on_press, on_release=self.on_release)
        self.listener.start()

    def on_press(self, key):
        if isinstance(key, KeyCode):
            k_char = key.char
            if k_char in self.keys:
                self.keys[k_char] = True

    def on_release(self, key):
        if isinstance(key, KeyCode):
            k_char = key.char
            if k_char in self.keys:
                self.keys[k_char] = False

    def get_act_from_input(self):
        act = [0, 0]
        if self.keys['i']:
            act[0] += 1
        if self.keys['k']:
            act[0] -= 1
        if self.keys['j']:
            act[1] += 1
        if self.keys['l']:
            act[1] -= 1
        return act

def predict_expected_cost(obs, action, classifier, regressor):
    data = np.append(obs, action)
    # remove features not used in the model namely "velocimeter2", "accelerometer2", "magnetometer2", "gyro0" and "gyro1"
    data = np.delete(data, [2, 5, 6, 7, 11])
    data = torch.from_numpy(data).float()
    data = data.unsqueeze(0)
    data = data.to(device)
    with torch.no_grad():
        # return zero if the sample is classified as 0 class and the prediction of the regressor otherwise
        if classifier(data).item() > 0.5:
            return regressor(data).item()
        else:
            return 0

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
classifier = torch.load("/home/user/bachelor/Safety-Gymnasium/classifier.pt")
classifier.to(device)
classifier.eval()
regressor = torch.load("/home/user/bachelor/Safety-Gymnasium/regressor.pt")
regressor.to(device)
regressor.eval()

env = safety_gymnasium.make('SafetyPointGoal1-v0', render_mode='human')
obs, info = env.reset()

input_act = InputAction()

terminated, truncated = False, False
ep_ret, ep_cost = 0, 0
while True:
    sleep(0.016)
    if terminated or truncated:
        print(f'Episode Return: {ep_ret} \t Episode Cost: {ep_cost}')
        ep_ret, ep_cost = 0, 0
        obs, info = env.reset()

    act = input_act.get_act_from_input()
    print(f"Expected cost: {predict_expected_cost(obs, act, classifier, regressor)}")
    obs, reward, cost, terminated, truncated, info = env.step(act)

    ep_ret += reward
    ep_cost += cost
