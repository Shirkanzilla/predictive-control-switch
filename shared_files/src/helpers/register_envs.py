import copy
import gymnasium
import safety_gymnasium
from omnisafe.envs.core import env_register, env_unregister
from helpers.saute_gridworlds import OmnisafeGridWorldsEnv
from helpers.saute_omnisafe_inverted_pendulum import OmnisafeSauteInvertedPendulumEnv
from helpers.saute_safety_gym_envs import OmnisafeSauteSafetyGymEnvs
from helpers.neural_shielding import NeuralShieldingEnv
from typing import Any, ClassVar

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
    gymnasium.register(id="SauteInvertedPendulum-v4",
        entry_point="helpers.saute_omnisafe_inverted_pendulum:SauteInvertedPendulumEnv",
        max_episode_steps=1000,
        reward_threshold=950.0,
    )

@env_register
@env_unregister
class SauteOmnisafeGridWorlds(OmnisafeGridWorldsEnv):
    pass

@env_register
@env_unregister
class SauteOmnisafeInvertedPendulum(OmnisafeSauteInvertedPendulumEnv):
    pass

@env_register
@env_unregister
class SauteOmnisafeSafetyGymEnvs(OmnisafeSauteSafetyGymEnvs):
    pass

@env_register
@env_unregister
class NeuralShieldingEnvs(NeuralShieldingEnv):
    def __init__(self, env_id:str, **kwargs) -> None:
        super(NeuralShieldingEnvs, self).__init__(env_id, **kwargs)

        