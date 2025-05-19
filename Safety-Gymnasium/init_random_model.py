from omnisafe.models.actor import GaussianLearningActor
import safety_gymnasium
import torch

env = safety_gymnasium.make('SafetyPointGoal1-v0', render_mode="")
obs_space = env.observation_space
act_space = env.action_space

randomActor = GaussianLearningActor(obs_space, act_space, [64,64])

observation, info = env.reset()

episode_over = False
while not episode_over:
    obs_tensor = torch.from_numpy(observation).float()
    action = randomActor.predict(obs_tensor).detach().numpy()
    observation, reward, terminated, truncated, info = env.step(action)

    episode_over = terminated or truncated

e