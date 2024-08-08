# Policy Gradient Methods
A collection of policy gradient methods to solve the mujoco environments


# Table of Contents

- [Results](#results)
- [How to install](#how-to-install)
- [How to run](#how-to-run)

# Results

### HalfCheetah-v4
<table style="width:100%;">
  <tr>
    <th></th>
    <th>Walker2D-v4</th>
    <th>HalfCheetah-v4</th>
    <th>Pusher-v4</th>
    <th>Ant-v4</th>
  </tr>
  <tr>
    <th>TD3</th>
    <td><img src="output/Walker2d-v4/240808-104135/fps_30_index_1000000_steps_1000_max_steps_500.gif" alt="Result 700k" style="width:100%;"/></td>
    <td><img src="output/HalfCheetah-v4/240808-012041/fps_30_index_550000_steps_1000_max_steps_250.gif" alt="Result 15k" style="width:100%;"/></td>
    <td><img src="output/Pusher-v4/240808-031547/fps_30_index_1000000_steps_400_max_steps_100.gif" alt="Result 15k" style="width:100%;"/></td>
    <td><img src="output/Pusher-v4/240808-025540/fps_30_index_1000000_steps_400_max_steps_100.gif" alt="Result 15k" style="width:100%;"/></td>
  </tr>
  <tr>
    <th>SAC</th>
    <td><img src="output/Walker2d-v4/240808-104135/fps_30_index_1000000_steps_1000_max_steps_500.gif" alt="Result 700k" style="width:100%;"/></td>
    <td><img src="output/HalfCheetah-v4/240808-012041/fps_30_index_550000_steps_1000_max_steps_250.gif" alt="Result 15k" style="width:100%;"/></td>
    <td><img src="output/Pusher-v4/240808-031547/fps_30_index_1000000_steps_400_max_steps_100.gif" alt="Result 15k" style="width:100%;"/></td>
    <td><img src="output/Pusher-v4/240808-025540/fps_30_index_1000000_steps_400_max_steps_100.gif" alt="Result 15k" style="width:100%;"/></td>
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
