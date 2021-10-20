# Optimal Adaptive Allocation using Deep Reinforcement Learning in a Dose-Response Study
Supplementary Materials for Kentaro Matsuura, Junya Honda, Imad El Hanafi, Takashi Sozu, Kentaro Sakamaki "Optimal Adaptive Allocation using Deep Reinforcement Learning in a Dose-Response Study" Statistics in Medicine 202x; (doi:xxxxx)

## How to Setup
We recommend using Linux or WSL on Windows, because the Ray package in Python is more stable on Linux. For example, in Ubuntu 20.04 (Python 3.8 was already installed), I was able to install the necessary packages with the following commands.

### Install Ray
```
sudo apt update
sudo apt upgrade
sudo apt install python3-pip
sudo pip3 install tensorflow numpy pandas gym
sudo apt install cmake
sudo pip3 install -U ray
sudo pip3 install 'ray[rllib]'
```

### Install R and RPy2
```
echo -e "\n## For R package"  | sudo tee -a /etc/apt/sources.list
echo "deb https://cloud.r-project.org/bin/linux/ubuntu $(lsb_release -cs)-cran40/" | sudo tee -a /etc/apt/sources.list
sudo apt-key adv --keyserver keyserver.ubuntu.com --recv-keys E298A3A825C0D65DFD57CBB651716619E084DAB9
sudo apt update
sudo apt install make g++ r-base
sudo apt install libxml2-dev libssl-dev libcurl4-openssl-dev
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

When we used `c2-standard-4`（vCPUx4, RAM16GB) on Google Cloud Platform, the learning was completed within a day.

### Simulate single trial
After the learning, we will obtain a checkpoint in `~/ray_results/PPO_MCPMod-v0_[datetime]-[xxx]/checkpoint-[yyy]/`. To simulate single trial using the obtained rule, please move the checkpoint files (`checkpoint` and `checkpoint.tune_metadata`) in the directory to `checkpoint/` in this repository, and rename the files as you like (see the example files). Then, please run `simulate-single-trial_RL-MAE.py` like:

```
python3 simulate-single-trial_RL-MAE.py
```
