from src.utils import *
from src.networks import *
from src.memory import *
from src.agents import *
from src.trainer import *


def _memory(config, state_dims, action_dims):
    buffer = REPLAYBUFFERS.get(config['type'].lower(), 'per')
    return buffer(**config['config'], state_dims=state_dims, action_dims=action_dims)
    

def _td3(config, env, test=False):
    DEVICE = None
    state_dims = env.observation_space.shape[0]
    action_dims = env.action_space.shape[0]
    max_action = env.action_space.high
    min_action = env.action_space.low
    
    if test:
        DEVICE = 'cpu'
        memory = None
        
    else:
        memory = _memory(config['memory'], state_dims, action_dims)
        
    agent = TD3(
        state_dims=state_dims,
        action_dims=action_dims,
        max_action=max_action,
        min_action=min_action,
        memory=memory,
        DEVICE=DEVICE,
        **config['algo']['config']       
    )
    
    return agent



def build(config, env, logdir, test):
    algo = config['algo']['type'].lower()
    agent = AGENTS.get(algo, None)
    if agent is None:
        raise ValueError(f"Algorithm {algo} not implemented")

    else:
        agent = agent(config, env, test)
        trainer = Trainer(
            env=env,
            logdir=f'{logdir}/{env.spec.id}',
            config=config,
            agent=agent,
            **config['trainer']
        )
        return trainer



AGENTS = dict(
    td3=_td3
)

REPLAYBUFFERS = dict(
    per=PER,
)
