# Single Python File
import os
from numpy.random import seed
import omnisafe
import sys
from helpers.register_envs import register_envs

register_envs()

if len(sys.argv) < 2:
    print("Usage: python record_video.py [save_dir]")
    sys.exit(1)

LOG_DIR = sys.argv[1]
evaluator = omnisafe.Evaluator(render_mode='rgb_array')
for item in os.scandir(os.path.join(LOG_DIR, 'torch_save')):
    if item.is_file() and item.name.split('.')[-1] == 'pt':
        if item.name.split('.')[0] == 'epoch-0':
            continue
        evaluator.load_saved(
            save_dir=LOG_DIR, model_name=item.name, width=256, height=256, render_mode='rgb_array'
        )
        evaluator.render(num_episodes=1)
        evaluator.evaluate(num_episodes=100)