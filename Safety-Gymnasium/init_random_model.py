from omnisafe.models.actor import GaussianLearningActor
import safety_gymnasium
import torch

env = safety_gymnasium.make('SafetyRacecarGoal1-v0', render_mode="human")
#env = safety_gymnasium.make('SafetyCarFormulaOne1-v0', render_mode="human")
obs_space = env.observation_space
act_space = env.action_space

randomActor = GaussianLearningActor(obs_space, act_space, [256,256,256,256,256], activation='relu', weight_initialization_mode="orthogonal")
observation, info = env.reset()

episode_over = False
while not episode_over:
    obs_tensor = torch.from_numpy(observation).float()
    action = randomActor.predict(obs_tensor, deterministic=True).detach().numpy()
    observation, reward, cost, terminated, truncated, info = env.step(action)
    episode_over = terminated or truncated

env.close()