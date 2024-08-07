import gymnasium as gym
from tqdm import tqdm
from gymnasium.wrappers import RecordEpisodeStatistics
from src.baseclasses import BaseAgent


class Trainer:
    def __init__(
        self, 
        env: gym.Env, 
        agent: BaseAgent, 
        save_freq: int, 
        total_training_steps: int, 
        max_episode_steps: int, 
        test_episodes: int, 
        test_max_episode_steps: int = None, 
        truncate: bool = True
        ) -> None:
        
        self.env = env
        self.agent = agent
        self.save_freq = save_freq
        self.truncate = truncate
        self.test_episodes = test_episodes
        self.max_episode_steps = max_episode_steps
        self.total_training_steps = total_training_steps
        self.test_max_episode_steps = test_max_episode_steps
    
    def train(self):
        env = RecordEpisodeStatistics(self.env)
        progress_bar = tqdm(range(self.total_training_steps), total=self.total_training_steps)
        done, truncated, episode, steps = False, False, 0, 0
        state, _ = env.reset()
        
        for step in progress_bar:
            steps += 1
            action = self.agent.select_action(state, greedy=False)
            next_state, reward, done, truncated, info = env.step(action)
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
            )

            state = next_state
            if done or (self.truncate and truncated):
                state, _ = env.reset()
                self.agent.update_offline()
                episode += 1
                steps = 0
            
            if step % self.save_freq == 0:
                metadata, frames = self.test(env=env)
    
    def test(self, env):
        env = gym.make(env.spec.id, max_episode_steps=self.test_max_episode_steps, render_mode='rgb_array')
        env = RecordEpisodeStatistics(env)
        frames = []
        metadata = dict()
        for episode in range(self.test_episodes):
            done, truncated = False, False
            state, _ = env.reset()
            frames.append(env.render())
            while not (done or truncated):
                action = self.agent.select_action(state, greedy=True)
                state, *_ = env.step(action)
                frames.append(env.render())
                
            info = _[0]['episode']
            metadata[episode] = info
        
        return metadata, frames