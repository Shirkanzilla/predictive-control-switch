# Single Python File
import os
import omnisafe
import omnisafe_inverted_pendulum as omnisafe_inverted_pendulum
from omnisafe.envs.core import env_register, env_unregister
import safety_gymnasium

safety_gymnasium.register(id="SafetyInvertedPendulum-v4",
    entry_point="omnisafe_inverted_pendulum:SafetyInvertedPendulumEnv",
    max_episode_steps=1000,
    reward_threshold=950.0,
)

@env_register
@env_unregister
class OmnisafeInvertedPendulum(omnisafe_inverted_pendulum.OmnisafeInvertedPendulumEnv):
    pass

env = OmnisafeInvertedPendulum("SafetyInvertedPendulum-v4")

# Just fill your experiment's log directory in here.
# Such as: ~/omnisafe/examples/runs/PPOLag-{SafetyPointGoal1-v0}/seed-000-2023-03-07-20-25-48
#LOG_DIR = './runs/PPOLag-{SafetyInvertedPendulum-v4}/seed-000-2026-01-21-12-50-04'
LOG_DIR = './runs/PPOLag-{SafetyInvertedPendulum-v4}/seed-000-2026-02-03-17-23-15'

evaluator = omnisafe.Evaluator(render_mode='rgb_array')
for item in os.scandir(os.path.join(LOG_DIR, 'torch_save')):
    if item.is_file() and item.name.split('.')[-1] == 'pt':
        evaluator.load_saved(
            save_dir=LOG_DIR, model_name=item.name, camera_name='track', width=256, height=256
        )
        evaluator.render(num_episodes=1)
        evaluator.evaluate(num_episodes=1)