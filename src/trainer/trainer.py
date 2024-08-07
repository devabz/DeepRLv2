import os
import json
import gymnasium as gym
from tqdm import tqdm
from gymnasium.wrappers import RecordEpisodeStatistics
from src.agents.base import BaseAgent
from src.utils import compile_to_mp4
from collections import defaultdict
from torch.utils.tensorboard import SummaryWriter
from datetime import datetime
from pathlib import Path
from multiprocessing import Process

def test(env_name, agent, test_max_episode_steps, test_episodes):
        env = gym.make(env_name, max_episode_steps=test_max_episode_steps, render_mode='rgb_array')
        env = RecordEpisodeStatistics(env)
        frames = []
        metadata = dict()
        for _ in range(test_episodes):
            done, truncated = False, False
            state, _ = env.reset()
            frames.append(env.render())
            step = 0
            while not (done or truncated):
                step += 1
                if step > test_max_episode_steps:
                    break
                
                action = agent.select_action(state, greedy=True)
                state, *_ = env.step(action)
                frames.append(env.render())
          
        return metadata, frames
    
def test_agent(path, video_path, env_name, agent_config, agent_class, max_episode_steps, test_episodes, fps):
    agent = agent_class(**agent_config)
    agent.load(path) 
    _, frames = test(env_name, agent, max_episode_steps, test_episodes)
    compile_to_mp4(frames, fps, video_path)


    
class Trainer:
    def __init__(
        self, 
        logdir:str,
        config: dict,
        env: gym.Env, 
        save_freq: int, 
        batch_size: int,
        update_freq: int,
        agent: BaseAgent, 
        test_episodes: int, 
        max_episode_steps: int, 
        total_training_steps: int, 
        test_max_episode_steps: int = None, 
        truncate: bool = True,
        record: bool = True,
        fps: int = 30,
        ) -> None:
        
        self.fps = fps
        self.env = env
        self.agent = agent
        self.record = record
        self.truncate = truncate
        self.save_freq = save_freq
        self.batch_size = batch_size
        self.update_freq = update_freq
        self.test_episodes = test_episodes
        self.max_episode_steps = max_episode_steps
        self.total_training_steps = total_training_steps
        self.test_max_episode_steps = test_max_episode_steps
        
        agent_name = self.agent.__class__.__name__.lower()
        self.logdir = Path(logdir)
        self.date_tag = datetime.now().strftime('%y%m%d-%H%M%S')
        self.artifacts_directory = self.logdir/agent_name/self.date_tag
        self.videos_directory = self.artifacts_directory/'videos'
        self.models_directory = self.artifacts_directory/'models'
        self.writer_directory = self.logdir/f'tensorboard'/agent_name/self.date_tag
        self.config = config
        os.makedirs(self.writer_directory, exist_ok=True)
        os.makedirs(self.artifacts_directory, exist_ok=True)
        with open(self.artifacts_directory/'config.json', 'w') as f:
            json.dump(config, f, indent=4)
        
        os.makedirs(self.videos_directory, exist_ok=True)
        os.makedirs(self.models_directory, exist_ok=True)
        self.writer: SummaryWriter = SummaryWriter(self.writer_directory)
        

    def train(self):
        env = RecordEpisodeStatistics(self.env)
        progress_bar = tqdm(range(self.total_training_steps), total=self.total_training_steps)
        done, truncated, episode, steps = False, False, 0, 0
        state, _ = env.reset()
        ep_rw = 0
        test_process = None
        for step in progress_bar:
            progress_bar.set_description_str(str(dict(episode=episode,  steps=steps, reward=ep_rw, updates=self.agent.updates, top=self.agent.memory.tree[0])))
            
            steps += 1
            action = self.agent.select_action(state, greedy=False)
            next_state, reward, done, truncated, info = env.step(action)
            ep_rw += reward
            self.agent.update_online(
                step=step, 
                steps=steps,
                episode=episode,
                state=state, 
                action=action, 
                reward=reward, 
                next_state=next_state, 
                done=done,
                truncated=truncated,
                batch_size=self.batch_size,
                update_freq=self.update_freq
            )

            state = next_state
            if done or (self.truncate and truncated):
                self.writer.add_scalar(f'0_reward', info['episode']['r'],  step)
                done, truncated = False, False
                state, _ = env.reset()
                self.agent.update_offline()
                episode += 1
                steps = 0
                ep_rw = 0
            
            if (step + 1) % self.save_freq == 0:
                path = self.models_directory/f'{step + 1}'
                os.makedirs(path, exist_ok=True)
                self.agent.save(path)
                test_process = self.run_test_process(env, step, path)
                    
        if test_process is not None:
            print(f'Ending process')
            test_process.join()
                
                    
    def run_test_process(self, env, step, path):
        if self.record:
            test_process = Process(target=test_agent, args=(
                path,
                self.videos_directory/f'{step + 1}.mp4',
                env.spec.id, 
                {
                    **self.config['algo']['config'], 
                    "memory": None, 
                    "state_dims": env.observation_space.shape[0], 
                    "action_dims": env.action_space.shape[0],
                    "min_action": env.action_space.low,
                    "max_action": env.action_space.high,
                    "DEVICE": 'cpu'
                }, 
                self.agent.__class__,
                self.test_max_episode_steps,
                self.test_episodes, 
                self.fps
            ))
            test_process.start()

            return test_process