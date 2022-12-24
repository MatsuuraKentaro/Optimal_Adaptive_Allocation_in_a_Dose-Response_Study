import numpy as np
import pandas as pd
import gym
import ray
from ray.rllib.agents.ppo import PPOTrainer, DEFAULT_CONFIG
import rpy2.robjects as robjects
import MCPMod
from ray.tune.registry import register_env
from MCPMod.envs.MCPModEnv import MCPModEnv

ENV_NAME = 'MCPMod-v0'
register_env(ENV_NAME, lambda config: MCPModEnv(config))
config = DEFAULT_CONFIG.copy()

## hyperparameter settings
config['seed'] = 123
config['gamma'] = 1.0
config['framework'] = 'torch'
config['num_workers'] = 4
config['num_sgd_iter'] = 20
config['num_cpus_per_worker'] = 1
config['sgd_minibatch_size'] = 200
config['train_batch_size'] = 10000

## simulation settings
config['env_config'] = {'reward_type':'score_MAE', 'model_type':'random', 'max_eff':1.65, 'alpha':0.025}

ray.init(ignore_reinit_error=True, log_to_driver=False)
agent = PPOTrainer(config, ENV_NAME)

N_update = 1000
results = []
episode_data = []

for n in range(1, N_update+1):
    result = agent.train()
    results.append(result)
    episode = {'n': n,
               'episode_reward_min':  result['episode_reward_min'],
               'episode_reward_mean': result['episode_reward_mean'],
               'episode_reward_max':  result['episode_reward_max'],
               'episode_len_mean':    result['episode_len_mean']}
    episode_data.append(episode)
    print(f'{n:3d}: Min/Mean/Max reward: {result["episode_reward_min"]:8.4f}/{result["episode_reward_mean"]:8.4f}/{result["episode_reward_max"]:8.4f}')
    if n >= 500 and n % 100 == 0:
        checkpoint_path = agent.save()
        print(checkpoint_path)

df = pd.DataFrame(data=episode_data)
df.to_csv('result_learn_RL-MAE.csv', index=False)
ray.shutdown()
