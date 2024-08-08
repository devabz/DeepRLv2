# Policy Gradient Methods
A collection of policy gradient methods to solve the mujoco environments


# Table of Contents

- [Results](#results)
- [How to install](#how-to-install)
- [How to run](#how-to-run)

# Results

### HalfCheetah-v4
<table>
  <tr>
    <th>15k Time Steps</th>
    <th>700k Time Steps</th>
  </tr>
  <tr>
    <td><img src="output/HalfCheetah-v4/240808-012041/fps_30_index_550000_steps_1000_max_steps_250.gif" alt="Result 15k" width="100"/></td>
    <td><img src="output/HalfCheetah-v4/240808-012041/fps_30_index_550000_steps_1000_max_steps_250.gif" alt="Result 700k" width="100"/></td>
  </tr>
</table>


# How to install
```
git clone https://github.com/devabz/DeepRLv2.git
cd DeepRLv2
conda env create --file environment.yml
conda activate DeepRLv2
```

# How to run
```
python main.py --config ./path/to/agent/config.json --logdir ./path/to/logdir 
```
### Example
```
python main.py --config templates/td3.json --logdir logs
```

# Components

## Algorithm
- Deep Delayed Policy Gradient (DDPG) 
- Twin Delayed DDPG (TD3)
- Soft Actor-Critic (SAC)
- Proximal Policy Optimization (PPO)

## Memory
- Experience Replay (ER)
- Prioritized Experience Replay (PER)
- Hindsight Experience Replay (HER)

## Trainer

# Optimizations
## Vectorized PER
## Parallelized Testing
