import json
import os

from typing import Any

from gymnasium.spaces import Box
import numpy as np

import torch

from omnisafe.common import Normalizer
from omnisafe.envs.wrapper import ActionRepeat, ActionScale, ObsNormalize, TimeLimit
from omnisafe.models.actor_critic.constraint_actor_q_critic import ConstraintActorQCritic
from omnisafe.utils.config import Config
from omnisafe.envs.core import CMDP, make
from omnisafe.algorithms.model_based.base.ensemble import EnsembleDynamicsModel
from omnisafe.models.actor import ActorBuilder
from typing import Dict, Tuple, Any


def _load_model_and_env(
    save_dir: str,
    model_name: str,
    cfgs: Config,
    env_kwargs: Dict[str, Any],
) -> None:
    """Load the model from the save directory.

    Args:
        save_dir (str): Directory where the model is saved.
        model_name (str): Name of the model.
        env_kwargs (dict[str, Any]): Keyword arguments for the environment.

    Raises:
        FileNotFoundError: If the model is not found.
    """
    # load the saved model
    model_path = os.path.join(save_dir, 'torch_save', model_name)
    try:
        model_params = torch.load(model_path)
    except FileNotFoundError as error:
        raise FileNotFoundError('The model is not found in the save directory.') from error

    # load the environment
    env = make(**env_kwargs)

    observation_space = env.observation_space
    action_space = env.action_space
    if 'Saute' in cfgs['algo'] or 'Simmer' in cfgs['algo']:
        safety_budget = (
            cfgs.algo_cfgs.safety_budget
            * (1 - cfgs.algo_cfgs.saute_gamma**cfgs.algo_cfgs.max_ep_len)
            / (1 - cfgs.algo_cfgs.saute_gamma)
            / cfgs.algo_cfgs.max_ep_len
            * torch.ones(1)
        )
    assert isinstance(observation_space, Box), 'The observation space must be Box.'
    assert isinstance(action_space, Box), 'The action space must be Box.'

    if cfgs['algo_cfgs']['obs_normalize']:
        obs_normalizer = Normalizer(shape=observation_space.shape, clip=5)
        obs_normalizer.load_state_dict(model_params['obs_normalizer'])
        env = ObsNormalize(env, device=torch.device('cpu'), norm=obs_normalizer)
    if env.need_time_limit_wrapper:
        env = TimeLimit(env, device=torch.device('cpu'), time_limit=1000)
    env = ActionScale(env, device=torch.device('cpu'), low=-1.0, high=1.0)

    if hasattr(cfgs['algo_cfgs'], 'action_repeat'):
        env = ActionRepeat(
            env,
            device=torch.device('cpu'),
            times=cfgs['algo_cfgs']['action_repeat'],
        )
    if hasattr(cfgs, 'algo') and cfgs['algo'] in [
        'LOOP',
        'SafeLOOP',
        'PETS',
        'CAPPETS',
        'RCEPETS',
        'CCEPETS',
    ]:
        dynamics_state_space = (
            env.coordinate_observation_space
            if env.coordinate_observation_space is not None
            else env.observation_space
        )
        assert env.action_space is not None and isinstance(
            env.action_space.shape,
            tuple,
        )
        if isinstance(env.action_space, Box):
            action_space = env.action_space
        else:
            raise NotImplementedError
        if cfgs['algo'] in ['LOOP', 'SafeLOOP']:
            actor_critic = ConstraintActorQCritic(
                obs_space=dynamics_state_space,
                act_space=action_space,
                model_cfgs=cfgs.model_cfgs,
                epochs=1,
            )
        if actor_critic is not None:
            actor_critic.load_state_dict(model_params['actor_critic'])
            actor_critic.to('cpu')
        dynamics = EnsembleDynamicsModel(
            model_cfgs=cfgs.dynamics_cfgs,
            device=torch.device('cpu'),
            state_shape=dynamics_state_space.shape,
            action_shape=action_space.shape,
            actor_critic=actor_critic,
            rew_func=None,
            cost_func=env.get_cost_from_obs_tensor,
            terminal_func=None,
        )
        dynamics.ensemble_model.load_state_dict(model_params['dynamics'])
        dynamics.ensemble_model.to('cpu')
        if cfgs['algo'] in ['CCEPETS', 'RCEPETS', 'SafeLOOP']:
            algo_to_planner = {
                'CCEPETS': (
                    'CCEPlanner',
                    {'cost_limit': cfgs['algo_cfgs']['cost_limit']},
                ),
                'RCEPETS': (
                    'RCEPlanner',
                    {'cost_limit': cfgs['algo_cfgs']['cost_limit']},
                ),
                'SafeLOOP': (
                    'SafeARCPlanner',
                    {
                        'cost_limit': cfgs['algo_cfgs']['cost_limit'],
                        'actor_critic': actor_critic,
                    },
                ),
            }
        elif cfgs['algo'] in ['PETS', 'LOOP']:
            algo_to_planner = {
                'PETS': ('CEMPlanner', {}),
                'LOOP': ('ARCPlanner', {'actor_critic': actor_critic}),
            }
        elif cfgs['algo'] in ['CAPPETS']:
            lagrange: torch.nn.Parameter = torch.nn.Parameter(
                model_params['lagrangian_multiplier'].to('cpu'),
                requires_grad=False,
            )
            algo_to_planner = {
                'CAPPETS': (
                    'CAPPlanner',
                    {
                        'cost_limit': cfgs['lagrange_cfgs']['cost_limit'],
                        'lagrange': lagrange,
                    },
                ),
            }
        planner_name = algo_to_planner[cfgs['algo']][0]
        planner_special_cfgs = algo_to_planner[cfgs['algo']][1]
        planner_cls = globals()[f'{planner_name}']
        planner = planner_cls(
            dynamics=dynamics,
            planner_cfgs=cfgs.planner_cfgs,
            gamma=float(cfgs.algo_cfgs.gamma),
            cost_gamma=float(cfgs.algo_cfgs.cost_gamma),
            dynamics_state_shape=dynamics_state_space.shape,
            action_shape=action_space.shape,
            action_max=1.0,
            action_min=-1.0,
            device='cpu',
            **planner_special_cfgs,
        )

    else:
        if 'Saute' in cfgs['algo'] or 'Simmer' in cfgs['algo']:
            observation_space = Box(
                low=np.hstack((observation_space.low, -np.inf)),
                high=np.hstack((observation_space.high, np.inf)),
                shape=(observation_space.shape[0] + 1,),
            )
        actor_type = cfgs['model_cfgs']['actor_type']
        pi_cfg = cfgs['model_cfgs']['actor']
        weight_initialization_mode = cfgs['model_cfgs']['weight_initialization_mode']
        actor_builder = ActorBuilder(
            obs_space=observation_space,
            act_space=action_space,
            hidden_sizes=pi_cfg['hidden_sizes'],
            activation=pi_cfg['activation'],
            weight_initialization_mode=weight_initialization_mode,
        )
        actor = actor_builder.build_actor(actor_type)
        actor.load_state_dict(model_params['pi'])

    return env, actor


def _load_cfgs(save_dir):
    cfg_path = os.path.join(save_dir, 'config.json')
    try:
        with open(cfg_path, encoding='utf-8') as file:
            kwargs = json.load(file)
    except FileNotFoundError as error:
        raise FileNotFoundError(
            f'The config file is not found in the save directory{save_dir}.',
        ) from error
    return Config.dict2config(kwargs)


# LOG_DIR should contain two things:
# 1. config.json
# 2. torch_save/{model_name}
#
# model_name usually looks like 'epoch-500.pt'
def load_guide(save_dir, model_name) -> Tuple[CMDP, ConstraintActorQCritic]:
    cfgs = _load_cfgs(save_dir)

    env_kwargs = {
        'env_id': cfgs['env_id'],
        'num_envs': 1,
    }

    env, actor = _load_model_and_env(save_dir, model_name, cfgs, env_kwargs)
    return env, actor
