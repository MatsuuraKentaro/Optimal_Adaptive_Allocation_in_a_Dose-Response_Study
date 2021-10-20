import rpy2.robjects as robjects
import numpy as np
import pandas as pd
import gym
from gym import spaces
from gym.utils import seeding

class MCPModEnv(gym.Env):
    def __init__(self, config):
        self.reward_type = config['reward_type']
        self.model_type  = config['model_type']
        self.max_eff     = config['max_eff']
        self.alpha       = config['alpha']
        self.D       = 5
        self.N_ini   = 50
        self.N_block = 10
        self.N_total = 150
        self.Delta   = 1.3
        self.SD      = np.sqrt(4.5)
        robjects.globalenv['max_eff'] = robjects.FloatVector(np.array([self.max_eff]))
        robjects.globalenv['alpha']   = robjects.FloatVector(np.array([self.alpha]))
        robjects.r('''
            suppressPackageStartupMessages(library('DoseFinding'))
            doses <- seq(from=0, to=8, by=2)   #  should be changed with self.D
            models_test <- Mods(doses = doses, maxEff = 1.65, linear = NULL, emax = 0.79, sigEmax = c(4, 5))
            mD <- max(doses)
            bnds <- list(emax=c(0.001, 0.5)*mD, sigEmax=matrix(c(0.001*mD, 2, mD, 6), 2))
            models <- Mods(doses = doses, maxEff = max_eff, linear = NULL, emax = 0.79, sigEmax = c(4, 5), exponential = 1, quadratic = - 1/12)
            resps <- getResp(models, doses=doses)
        ''')
        self.action_space = spaces.Discrete(self.D)
        self.observation_space = spaces.Box(
            low = np.hstack([
                np.repeat(-np.finfo(np.float32).max, self.D-1),
                np.repeat(0.0, self.D),
                np.repeat(0.0, self.D)
            ]),
            high = np.hstack([
                np.repeat(np.finfo(np.float32).max, self.D-1),
                np.repeat(np.finfo(np.float32).max, self.D),
                np.repeat(1.0, self.D)
            ]),
            dtype=np.float32
        )
        self.seed()

    def seed(self, seed=None):
        self.np_random, seed = seeding.np_random(seed)
        return [seed]

    def step(self, action, action_array=False):
        if action_array:
            doses_draw = action
        else:
            doses_draw = [action] * self.N_block

        resps_draw = self.np_random.normal(self.resps_true[doses_draw], self.SD).tolist()
        self.doses.extend(doses_draw)
        self.resps.extend(resps_draw)

        if len(self.doses) >= self.N_total:
            done = True
            robjects.globalenv['sim_doses_idx'] = robjects.IntVector(np.array(self.doses) + 1)
            robjects.globalenv['sim_resps']     = robjects.FloatVector(self.resps)
            robjects.r('''
                set.seed(1)
                sim_doses <- doses[sim_doses_idx]
                suppressMessages(resmm <- MCPMod(sim_doses, sim_resps, models=models_test, Delta=delta, alpha=alpha, selModel='AIC', bnds=bnds))
                if (is.null(resmm$selMod)) {
                  suppressMessages(resmm <- MCPMod(sim_doses, sim_resps, models=models_test, Delta=delta, alpha=1.0, selModel='AIC', bnds=bnds))
                }
                selmod <- resmm$selMod
                pvals <- attr(resmm$MCTtest$tStat, 'pVal')
                pval <- min(pvals)
                med <- unname(resmm$doseEst[selmod])
                resps_est <- predict(resmm$mods[[selmod]], predType='ls-means', doseSeq=doses)
                score_power <- ifelse(pval < alpha, 1, 0)
                score_MS    <- ifelse(selmod == model_name, 1, 0)
                score_TD    <- ifelse(!is.na(med) & med_range[1] < med & med < med_range[2], 1, 0)
                score_MAE   <- 1 - 2*mean(abs((resps_est - resps_est[1]) - resps_true)[-1])
            ''')
            pval   = robjects.r['pval'][0]
            selmod = robjects.r['selmod'][0]
            med    = robjects.r['med'][0]
            score_power = robjects.r['score_power'][0]
            score_MS    = robjects.r['score_MS'][0]
            score_TD    = robjects.r['score_TD'][0]
            score_MAE   = robjects.r['score_MAE'][0]

            info = {'pval': pval, 'selmod': selmod, 'med': med, 'score_power':score_power, 'score_MS':score_MS, 'score_TD':score_TD, 'score_MAE':score_MAE}
            reward = info[self.reward_type] if self.reward_type in info else 0
        else:
            done = False
            info = {}
            reward = 0
        info.update(**{'dose': self.doses, 'resp': self.resps})
        return self._get_obs(), reward, done, info

    def _get_obs(self):
        df = pd.DataFrame(self.resps).groupby(self.doses)
        means       = df.mean().values.flatten()
        diffs       = means[1:] - means[0]
        counts      = df.count().values.flatten()
        stds        = df.std(ddof=0).values.flatten()
        draw_ratios = counts / self.N_total
        return np.hstack([diffs, stds, draw_ratios])

    def reset(self):
        if self.model_type == 'random':
            self.model_name = self.np_random.choice(['linear', 'emax', 'sigEmax'])
        else:
            self.model_name = self.model_type
        robjects.globalenv['model_name'] = robjects.StrVector([self.model_name])
        robjects.globalenv['delta']      = robjects.FloatVector([self.Delta])
        robjects.r('''
            if (model_name == 'flat') {
              resps_true <- rep(0, length(doses))
              med_range <- c(0, 0)
            } else {
              resps_true <- resps[, model_name]
              med_lower <- unname(TD(models, Delta=delta*0.9)[model_name])
              med_upper <- unname(TD(models, Delta=delta*1.1)[model_name])
              med_upper <- ifelse(!is.na(med_upper) & med_upper <= mD, med_upper, mD)
              med_range <- c(med_lower, med_upper)
            }
        ''')
        self.resps_true = np.array(robjects.r['resps_true'])
        self.doses = np.repeat([np.arange(self.D)], self.N_ini / self.D).tolist()
        self.resps = self.np_random.normal(self.resps_true[self.doses], self.SD).tolist()
        return self._get_obs()
