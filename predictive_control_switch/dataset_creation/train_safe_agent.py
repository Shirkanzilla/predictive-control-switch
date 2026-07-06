import omnisafe
from helpers.register_envs import register_envs
import sys
import pickle
import json
register_envs()

# Train
if len(sys.argv) < 4:
    print("Usage: python train_safe_agent.py [env_id] [algorithm] [epochs] [device] [threads] [curriculum_learning y/n] [curriculum_learning_epochs] [previous_model_path (optional)]")
    sys.exit(1)

env_id = sys.argv[1]
algorithm = sys.argv[2]
epochs = sys.argv[3]
device = sys.argv[4] if len(sys.argv) > 4 else "cpu"
threads = sys.argv[5] if len(sys.argv) > 5 else 20
curriculum_learning = sys.argv[6].lower() == 'y' if len(sys.argv) > 6 else False
curriculum_learning_epochs = sys.argv[7] if len(sys.argv) > 7 else 50
previous_model_path = sys.argv[8] if len(sys.argv) > 8 else None

# Train for 50 epochs (default steps is 20.000, so i lowered the total steps from 10.000.000 to 1.000.000) https://github.com/PKU-Alignment/omnisafe/blob/main/omnisafe/configs/on-policy/PPOLag.yaml for reference
custom_cfgs = {
    'train_cfgs': {
        'torch_threads': int(threads), 
        'total_steps': 20000 * int(epochs),
        'device' : device,
    },
}

if "FrozenLake" in env_id:
    custom_cfgs["algo_cfgs"] = {"obs_normalize": False}

if previous_model_path is None:
    env_id = f"SafeAgentBase{env_id}" if curriculum_learning else env_id
    agent = omnisafe.Agent(algorithm, env_id=env_id, custom_cfgs=custom_cfgs)
    agent.learn()
    agent.plot(smooth=1)
    state_dict = agent.agent._actor_critic.state_dict()
    with open(f"{agent.agent.logger.log_dir}/full_state_dict.pkl", "wb") as f:
        pickle.dump(agent.agent._actor_critic.state_dict(), f)

if curriculum_learning:
    custom_cfgs["train_cfgs"]["total_steps"] = 20000 * int(curriculum_learning_epochs)
    env_id = env_id.replace("Base", "Advanced")
    agent = omnisafe.Agent(algorithm, env_id=env_id, custom_cfgs=custom_cfgs)
    if previous_model_path is None:
        state_dict = agent.agent._actor_critic.state_dict()
        agent.agent._actor_critic.load_state_dict(state_dict)
    else:
        with open(previous_model_path, "rb") as f:
            state_dict = pickle.load(f)
        agent.agent._actor_critic.load_state_dict(state_dict)
    # continue training with harder dynamics
    agent.learn()
    agent.plot(smooth=1)
