# Optimal Adaptive Allocation using Deep Reinforcement Learning in a Dose-Response Study
Supplementary Materials for Kentaro Matsuura, Junya Honda, Imad El Hanafi, Takashi Sozu, Kentaro Sakamaki "Optimal Adaptive Allocation using Deep Reinforcement Learning in a Dose-Response Study" Statistics in Medicine 2021; (doi:10.1002/sim.9247)

## How to Setup
We recommend using Linux or WSL on Windows, because the Ray package in Python is more stable on Linux. For example, in Ubuntu 20.04 (Python 3.8 was already installed), I was able to install the necessary packages with the following commands.

### Install Ray
```
sudo apt update
sudo apt upgrade
sudo apt install python3-pip
sudo pip3 install torch
sudo pip3 install -U ray
```

### Install R and RPy2
To install R, see https://cran.r-project.org/bin/linux/ubuntu/

```
sudo pip3 install rpy2
```

### Install `DoseFinding` package in R
```
install.packages('DoseFinding')
```

## How to Use
### Change simulation settings
To change the simulation settings, it is necessary to understand `MCPMod/envs/MCPModEnv.py`. This part is a bit difficult because of the interaction between R and Python. Therefore, we have a plan to create an R package to use our method easily.

### Obtain adaptive allocation rule
To obtain RL-MAE by learning, please run `learn_RL-MAE.py` like:

```
nohup python3 learn_RL-MAE.py > std.log 2> err.log &
```

To obtain other RL-methods, please change the `reward_type` in line 25 in `learn_RL-MAE.py` to something like `score_TD`, then run the modified file.

When we used `c2-standard-4` (vCPUx4, RAM16GB) on Google Cloud Platform, the learning was completed within a day.

### Simulate single trial
After the learning, we will obtain a checkpoint in `~/ray_results/PPO_MCPMod-v0_[datetime]-[xxx]/checkpoint-[yyy]/`. To simulate single trial using the obtained rule, please move the checkpoint directory (`checkpoint-[yyy]`) to `checkpoint/` in this repository, and edit the path in `simulate-single-trial_RL-MAE.py`. Then, please run it:

```
python3 simulate-single-trial_RL-MAE.py
```
