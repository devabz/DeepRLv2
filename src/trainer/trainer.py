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
from multiprocessing import Process, Queue, queues


class Worker(Process):
    def __init__(self, task_queue):
        self.task_queue = task_queue
        super().__init__()
    
    def run(self):
        while True:
            try:
                task = self.task_queue.get()
                if task is None:
                    return

                func, *args = task
                func(*args)
                
            except queues.Empty:
                continue
        


def test(env_name, agent, test_max_episode_steps, test_episodes, path):
        env = gym.make(env_name, max_episode_steps=test_max_episode_steps, render_mode='rgb_array')
        env = RecordEpisodeStatistics(env)
        frames = []
        metadata = dict()
        total = test_episodes*test_max_episode_steps
        bar = tqdm(range(total), total=total)
    
        done, truncated = False, False
        state, _ = env.reset()
        for step in bar:
            frames.append(env.render())
            
            bar.set_description_str(f'{os.getpid()} - {path} - {(step // test_max_episode_steps ) + 1, step + 1}')
            
            action = agent.select_action(state, greedy=True)
            state, _, done, truncated, _ = env.step(action)
            
            if done or truncated:
                done, truncated = False, False
                state, _ = env.reset()
            
        frames.append(env.render())
        
        return metadata, frames
    
def test_agent(path, video_path, env_name, agent_config, agent_class, max_episode_steps, test_episodes, fps):
    
    #print(os.getpid(), f'triggered - {video_path}')
    agent = agent_class(**agent_config)
    agent.load(path) 
    _, frames = test(env_name, agent, max_episode_steps, test_episodes, path)
    compile_to_mp4(frames, fps, video_path)

    
class Trainer:
    def __init__(
        self, 
        logdir:str,
        config: dict,
        env: gym.Env, 
        save_total: int, 
        batch_size: int,
        update_freq: int,
        agent: BaseAgent, 
        test_episodes: int, 
        max_episode_steps: int, 
        start_select_actions: int,
        total_training_steps: int, 
        test_max_episode_steps: int = None, 
        
        truncate: bool = True,
        record: bool = True,
        fps: int = 30,
        workers: int = 4,
        ) -> None:
        
        self.fps = fps
        self.env = env
        self.agent = agent
        self.record = record
        self.config = config
        self.truncate = truncate
        self.save_total = save_total
        self.save_freq = total_training_steps//save_total
        self.batch_size = batch_size
        self.update_freq = update_freq
        self.test_episodes = test_episodes
        self.max_episode_steps = max_episode_steps
        self.total_training_steps = total_training_steps
        self.start_select_actions = start_select_actions
        self.test_max_episode_steps = test_max_episode_steps
        
        
        agent_name = self.agent.__class__.__name__.lower()
        self.date_tag = datetime.now().strftime('%y%m%d-%H%M%S')
        
        self.logdir = Path(logdir)
        self.writer_directory = self.logdir/f'tensorboard'/agent_name/self.date_tag
        self.artifacts_directory = self.logdir/agent_name/self.date_tag
        self.videos_directory = self.artifacts_directory/'videos'
        self.models_directory = self.artifacts_directory/'models'
        
        self.env = gym.make(self.env.spec.id, max_episode_steps=max_episode_steps)
        os.makedirs(self.writer_directory, exist_ok=True)
        os.makedirs(self.artifacts_directory, exist_ok=True)
        with open(self.artifacts_directory/'config.json', 'w') as f:
            json.dump(config, f, indent=4)
        
        os.makedirs(self.videos_directory, exist_ok=True)
        os.makedirs(self.models_directory, exist_ok=True)
        self.writer: SummaryWriter = SummaryWriter(self.writer_directory)
        
        self.task_queue = Queue()
        self.workers = [Worker(self.task_queue) for _ in range(workers)]
        for worker in self.workers:
            worker.start()
            

    def train(self):
        env = RecordEpisodeStatistics(self.env)
        progress_bar = tqdm(range(self.total_training_steps), total=self.total_training_steps)
        done, truncated, episode, steps = False, False, 0, 0
        state, _ = env.reset()
        ep_rw = 0
        test_process = None
        run_test = True
        for step in progress_bar:
            progress_bar.set_description_str(str(dict(env=env.spec.id, episode=episode,  steps=steps, reward=round(ep_rw, 2), updates=self.agent.updates, top=round(self.agent.memory.tree[0], 2))))
            
            steps += 1
            
            if step >= self.start_select_actions:
                action = self.agent.select_action(state, greedy=False)
            else:
                action = env.action_space.sample()
                
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
                self.run_test_process(env, step, path)
                        
                        
        self.shutdown_workers()
            
    def shutdown_workers(self):
        for worker in self.workers:
            self.task_queue.put(None)
        
        for worker in self.workers:
            worker.join()
            
                    
                    
    def run_test_process(self, env, step, path):
        if self.record:
            self.task_queue.put((
                test_agent,
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
            