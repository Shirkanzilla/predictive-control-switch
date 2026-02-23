import copy
import gymnasium
import safety_gymnasium
from omnisafe.envs.core import env_register, env_unregister
from helpers.omnisafe_inverted_pendulum import OmnisafeInvertedPendulumEnv
from helpers.saute_omnisafe_inverted_pendulum import OmnisafeSauteInvertedPendulumEnv
from helpers.saute_safety_point_goal import OmnisafeSauteGoalLevel1

def __register_helper(env_id, entry_point, spec_kwargs=None, **kwargs):
    """Register a environment to both Safety-Gymnasium and Gymnasium registry."""
    env_name, dash, version = env_id.partition('-')
    if spec_kwargs is None:
        spec_kwargs = {}

    safety_gymnasium.register(
        id=env_id,
        entry_point=entry_point,
        kwargs=spec_kwargs,
        **kwargs,
    )
    gymnasium.register(
        id=f'{env_name}Gymnasium{dash}{version}',
        entry_point='safety_gymnasium.wrappers.gymnasium_conversion:make_gymnasium_environment',
        kwargs={'env_id': f'{env_name}Gymnasium{dash}{version}', ** copy.deepcopy(spec_kwargs)},
        **kwargs,
    )

def register_envs():
    gymnasium.register(id="SafetyInvertedPendulum-v4",
        entry_point="helpers.omnisafe_inverted_pendulum:SafetyInvertedPendulumEnv",
        max_episode_steps=1000,
        reward_threshold=950.0,
    )
    gymnasium.register(id="SauteSafetyInvertedPendulum-v4",
        entry_point="helpers.saute_omnisafe_inverted_pendulum:SauteSafetyInvertedPendulumEnv",
        max_episode_steps=1000,
        reward_threshold=950.0,
    )
    env_id = 'SautePointGoal1-v0'
    combined_config = {}
    combined_config.update({'agent_name': 'Point'})

    __register_helper(
        env_id=env_id,
        entry_point='safety_gymnasium.builder:Builder',
        spec_kwargs={'config': combined_config, 'task_id': env_id},
        max_episode_steps=1000,
    )
    
@env_register
@env_unregister
class OmnisafeInvertedPendulum(OmnisafeInvertedPendulumEnv):
    pass

@env_register
@env_unregister
class SauteOmnisafeInvertedPendulum(OmnisafeSauteInvertedPendulumEnv):
    pass

@env_register
@env_unregister
class SauteGoalLevel1Env(OmnisafeSauteGoalLevel1):
    pass