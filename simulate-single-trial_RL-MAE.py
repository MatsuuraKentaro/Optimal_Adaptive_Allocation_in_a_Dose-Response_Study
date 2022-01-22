import numpy as np
import pandas as pd
from scipy.special import softmax
import gym
import ray
from ray.rllib.agents.ppo import PPOTrainer, DEFAULT_CONFIG
import rpy2.robjects as robjects
import MCPMod
from ray.tune.registry import register_env
from MCPMod.envs.MCPModEnv import MCPModEnv

ENV_NAME = 'MCPMod-v0'
register_env(ENV_NAME, lambda config: MCPModEnv(config))
ray.init(ignore_reinit_error=True, log_to_driver=False)

D = 5          # num of doses
N_block = 10   # num of subjects in a block
N_total = 150  # num of total subjects
len_state = (D-1) + D + D
reward_name = 'score_MAE'
alpha = 0.0165  # after adjustment
checkpoint_path = 'checkpoint/checkpoint_RL-MAE'
model_true = 'linear'
max_eff_true = 1.65
score_names = ['pval', 'selmod', 'med', 'score_power', 'score_MS', 'score_TD', 'score_MAE']
state_names = ['state' + str(i).zfill(2) for i in range(len_state)]
prob_names = ['prob' + str(i) for i in range(D)]

config = DEFAULT_CONFIG.copy()
config['seed'] = 1234
config['num_workers'] = 1
config['num_cpus_per_worker'] = 1
sim_config = {'reward_type': reward_name, 'model_type':model_true, 'max_eff':max_eff_true, 'alpha':alpha}
config['env_config'] = sim_config
agent = PPOTrainer(config, ENV_NAME)
env = gym.make(ENV_NAME, config = sim_config)
agent.restore(checkpoint_path)

results_score = []
results_data  = []
results_pi    = []

simID = 123
env.seed(simID)
state = env.reset()
done = False
blockID = 1
while not done:
    action = agent.compute_action(state, full_fetch=True)
    probs = softmax(action[2]['action_dist_inputs'])
    actions = np.random.choice(D, size=N_block, p=probs)
    results_pi.append([model_true, max_eff_true, simID, blockID, *probs, *state])
    state, reward, done, info = env.step(actions, action_array=True)
    blockID += 1
    if done:
        scores = [info[score_name] for score_name in score_names]
        results_score.append([model_true, max_eff_true, simID, *scores, *state])
        results_data += list(zip([model_true] * N_total,
                                 [max_eff_true] * N_total,
                                 [simID] * N_total,
                                 range(1, N_total + 1),
                                 np.array(info['dose']) + 1,
                                 np.array(info['resp'])))

df_score = pd.DataFrame(results_score, columns=['model_true', 'max_eff', 'simID', *score_names, *state_names])
df_data  = pd.DataFrame(results_data,  columns=['model_true', 'max_eff', 'simID', 'subjID', 'dose', 'resp'])
df_pi    = pd.DataFrame(results_pi,    columns=['model_true', 'max_eff', 'simID', 'blockID', *prob_names, *state_names])
df_score.to_csv('result_score.csv', index=False)
df_data.to_csv('result_data.csv', index=False)
df_pi.to_csv('result_pi.csv', index=False)

ray.shutdown()
