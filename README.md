# Policy Gradient Methods
A collection of policy gradient methods to solve the mujoco environments



# Table of Contents

- [How to install](#how-to-install)
- [How to run](#how-to-run)
- [Results](#results)

## Run using Docker
##### Download image 
```
...
```
##### Run image
```
...
```
##### Example
```
...

```
## Run Locally
##### Install locally
```
git clone https://github.com/devabz/DeepRLv2.git
cd DeepRLv2
conda env create --file environment.yml
conda activate DeepRLv2
```
##### Run locally
```
python train.py --config ./path/to/agent/config.json --logdir ./path/to/logdir [--env]
```
##### Example
```
python train.py --env Walker2d-v4 --config templates/td3.json --logdir logs
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




# Results

<table style="width:100%;">
  <tr>
    <th></th>
    <th>DDPG</th>
    <th>TD3</th>
    <th>SAC</th>
    <th>PPO</th>
  </tr>
  <tr>
    <th>TD3</th>
    <td><img src="gifs/td3/pusher-v4_2.gif" alt="Result 15k" style="width:100%; max-height:200px;" /></td>
    <td><img src="gifs/td3/hopper-4.gif" alt="Result 15k" style="width:100%; max-height:200px;" /></td>
    <td><img src="gifs/td3/humanoid-v4.gif" alt="Result 15k" style="width:100%; max-height:200px;" /></td>
    <td><img src="gifs/td3/walker2d-v4.gif" alt="Result 700k" style="width:100%; max-height:200px;" /></td>
    <td><img src="gifs/td3/ant-v4.gif" alt="Result 700k" style="width:100%; max-height:200px;" /></td>
    <td><img src="gifs/td3/halfcheetah-v4.gif" alt="Result 15k" style="width:100%; max-height:200px;" /></td>
  </tr>
</table>

