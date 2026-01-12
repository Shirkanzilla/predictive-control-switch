import os
# bugfix for my setup
os.environ["DISPLAY"] = ":0"
from pynput.keyboard import Listener, KeyCode
import safety_gymnasium
import gymnasium
from time import sleep
    
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

    def get_act_from_input(self, robot):
        act = []
        if robot == "Point":
            act = [0, 0]
            if self.keys['i']:
                act[0] += 1
            if self.keys['k']:
                act[0] -= 1
            if self.keys['j']:
                act[1] += 1
            if self.keys['l']:
                act[1] -= 1
        elif robot == "InvertedPendulum":
            act = [0]
            if self.keys['j']:
                act[0] = -1
            if self.keys['l']:
                act[0] = 1
        return act
#robot = "Point"
#env = safety_gymnasium.make(f'Safety{robot}FormulaOne1-v0', render_mode='human')
robot = "InvertedPendulum"
safety_gymnasium.register(id="CustomInvertedPendulum-v4",
    entry_point="custom_inverted_pendulum_v4:CustomInvertedPendulumEnv",
    max_episode_steps=1000,
    reward_threshold=950.0,
)
env = safety_gymnasium.make('CustomInvertedPendulum-v4', render_mode="human")
obs, info = env.reset()

input_act = InputAction()

terminated, truncated = False, False
ep_ret, ep_cost = 0, 0
while True:
    #sleep(0.016)
    sleep(0.16)
    if terminated or truncated:
        print(f'Episode Return: {ep_ret} \t Episode Cost: {ep_cost}')
        ep_ret, ep_cost = 0, 0
        obs, info = env.reset()

    #print(env.action_space.sample())
    act = input_act.get_act_from_input(robot)
    obs, reward, cost, terminated, truncated, info = env.step(act)
    #bs, reward, cost, terminated, truncated, info = env.step(act)
    print(f"Action: {act}, Reward: {reward}, Cost: {cost}")

    ep_ret += reward
    ep_cost += cost
