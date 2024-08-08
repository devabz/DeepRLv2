import os
import json
import gymnasium as gym
from pathlib import Path
from argparse import ArgumentParser
from gymnasium.wrappers import RecordEpisodeStatistics
from tqdm import tqdm
from src import *
from moviepy.editor import VideoFileClip

parser = ArgumentParser()
parser.add_argument('logdir', type=str, help='Log direectory')
parser.add_argument('output', type=str, help='Path to ouput directory')
parser.add_argument('--index', type=int, help='Model index', default=None)
parser.add_argument('--steps', type=int, help='Total steps', default=int(1e3))
parser.add_argument('--max_steps', type=int, help='Max episode steps', default=None)


def load(args):
    path = Path(args.logdir)
    with open(path/'config.json', 'r') as f:
        config = json.load(f)
    
    env = path.absolute().parent.parent.name
    env = gym.make(env, max_episode_steps=args.max_steps, render_mode='rgb_array')
    
    agent = AGENTS[config['algo']['type'].lower()](config=config, env=env, test=True)
    
    index = args.index
    models = path/'models'
    
    if index is None:
        index = sorted(os.listdir(models), key=int)[-1]
    
    models /= index
    
    agent.load(models)
    
    return agent, env, index, path, config


def convert_to_gif(path):
    # Load the MP4 file
    video = VideoFileClip(path)

    # Write the GIF to a file
    video.write_gif(path.replace('mp4', 'gif'))


if __name__ == "__main__":
    args = parser.parse_args()
    agent, env, model_index, path, config = load(args)
    name = path.absolute().name
    
    state_dims = env.observation_space.shape[0]
    action_dims = env.action_space.shape[0]

    
    
    max_steps = args.max_steps
    max_steps = env._max_episode_steps if max_steps is None else max_steps
    env = RecordEpisodeStatistics(env)
    state, _ = env.reset()
    done, truncated = False, False
    bar = tqdm(range(args.steps), total=args.steps)
    
    frames = []
    for step in bar:
        
        frames.append(env.render())
        
        state, reward, done, truncated, info = env.step(
            agent.select_action(state, greedy=True)
        )
        
        if (done or truncated):
            bar.set_description_str(f"reward: {info['episode']['r']}")
            state, _ = env.reset()
            done, truncated = False, False


    env.close()
    
    fps = 30
    
    path = Path(args.output)
    os.makedirs(path, exist_ok=True)
    path = path/env.spec.id/name
    os.makedirs(path, exist_ok=True)
    path=path/f'fps_{fps}_index_{model_index}_steps_{args.steps}_max_steps_{max_steps}.mp4'
    compile_to_mp4(frames, fps=fps, path=path)
    convert_to_gif(path.absolute().__str__())
    os.remove(path)
    