# Single Python File
import os
import omnisafe

# Just fill your experiment's log directory in here.
# Such as: ~/omnisafe/examples/runs/PPOLag-{SafetyPointGoal1-v0}/seed-000-2023-03-07-20-25-48
LOG_DIR = './runs/PPOLag-{SafetyPointGoal1-v0}/seed-000-2025-05-13-17-51-08'
evaluator = omnisafe.Evaluator(render_mode='rgb_array')
for item in os.scandir(os.path.join(LOG_DIR, 'torch_save')):
    if item.is_file() and item.name.split('.')[-1] == 'pt':
        evaluator.load_saved(
            save_dir=LOG_DIR, model_name=item.name, camera_name='track', width=256, height=256
        )
        evaluator.render(num_episodes=1)
        evaluator.evaluate(num_episodes=1)