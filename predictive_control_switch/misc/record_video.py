# Single Python File
import os
import omnisafe
from helpers.omnisafe_inverted_pendulum import OmnisafeInvertedPendulumEnv
from helpers.saute_omnisafe_inverted_pendulum import OmnisafeSauteInvertedPendulumEnv
from helpers.saute_safety_point_goal import OmnisafeSauteGoalLevel1
from omnisafe.envs.core import env_register, env_unregister
import safety_gymnasium
import gymnasium
import copy

from helpers.register_envs import register_envs

register_envs()

#env = OmnisafeInvertedPendulum("SauteSafetyInvertedPendulum-v4")

# Just fill your experiment's log directory in here.
# Such as: ~/omnisafe/examples/runs/PPOLag-{SafetyPointGoal1-v0}/seed-000-2023-03-07-20-25-48
#LOG_DIR = './runs/PPOLag-{SafetyInvertedPendulum-v4}/seed-000-2026-01-21-12-50-04'
#LOG_DIR = './runs/PPO-{SafetyInvertedPendulum-v4}/seed-000-2026-02-16-11-45-26'
LOG_DIR = './runs/PPO-{SauteSafetyInvertedPendulum-v4}/100epochs_gradual_cf'
evaluator = omnisafe.Evaluator(render_mode='rgb_array')
for item in os.scandir(os.path.join(LOG_DIR, 'torch_save')):
    if item.is_file() and item.name.split('.')[-1] == 'pt':
        evaluator.load_saved(
            save_dir=LOG_DIR, model_name=item.name, camera_name='track', width=256, height=256, render_mode='rgb_array'
        )
        evaluator.render(num_episodes=1)
        evaluator.evaluate(num_episodes=1)