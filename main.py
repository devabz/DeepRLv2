import gymnasium as gym

from src import *


if __name__ == "__main__":
    args = parser.parse_args()
    
    # Configuration
    env_name = 'HalfCheetah-v4'
    max_episode_steps = None
    hidden_dims = 256
    tau = 0.005
    lr_a = 0.001
    lr_c = 0.001
    gamma = 0.95
    noise = 0.25
    noise_clip = 0.25
    noise_decay = 0.999
    policy_delay = 2
    
    start_select_actions = 100_000
    
    workers = 4
    buffer_size = 50_000
    start_memory_updates = 2000
    batch_size = 2048
    save_total = 20
    test_episodes = 2
    test_max_episode_steps = 250
    
    update_freq = 25
    max_episode_steps = 500
    total_training_steps = 1_000_000
    save_freq = total_training_steps // save_total
    truncate = True
    record = True
    fps = 30
    priority_percentage = 0.25
    
    
    config = dict(
        algo=dict(
            type='TD3',
            config=dict(
                hidden_dims=hidden_dims,
                noise=noise,
                noise_clip=noise_clip,
                noise_decay=noise_decay,
                policy_delay=policy_delay,
                lr_a=lr_a,
                lr_c=lr_c,
                tau=tau,
                gamma=gamma
            )
        ),
        memory=dict(
            type='PER',
            config=dict(
                buffer_size=buffer_size,
                priority_percentage=priority_percentage,
                start_memory_updates=start_memory_updates
            )
        ),
        trainer=dict(
            save_total=save_total,
            save_freq=save_freq,
            batch_size=batch_size,
            update_freq=update_freq,
            test_episodes=test_episodes,
            max_episode_steps=max_episode_steps,
            total_training_steps=total_training_steps,
            test_max_episode_steps=test_max_episode_steps,
            truncate=truncate,
            record=record,
            start_select_actions=start_select_actions,
        )
    )
    
    # Instatiation
    env = gym.make(env_name, max_episode_steps=max_episode_steps)
    state_dims = env.observation_space.shape[0]
    action_dims = env.action_space.shape[0]
    
    memory = PER(
        state_dims=state_dims, 
        action_dims=action_dims,
        buffer_size=buffer_size,
        priority_percentage=priority_percentage,
        start_memory_updates=start_memory_updates
    )
    agent = TD3(
        state_dims=state_dims,
        action_dims=action_dims,
        hidden_dims=hidden_dims,
        noise=noise,
        noise_clip=noise_clip,
        noise_decay=noise_decay,
        policy_delay=policy_delay,
        min_action=env.action_space.low,
        max_action=env.action_space.high,
        lr_a=lr_a,
        lr_c=lr_c,
        tau=tau,
        gamma=gamma,
        memory=memory
    )
    
    trainer = Trainer(
        env=env,
        logdir=f'{args.logdir}/{env_name}',
        config=config,
        save_freq=save_freq,
        batch_size=batch_size,
        update_freq=update_freq,
        agent=agent,
        test_episodes=test_episodes,
        max_episode_steps=max_episode_steps,
        total_training_steps=total_training_steps,
        test_max_episode_steps=test_max_episode_steps,
        start_select_actions=start_select_actions,
        truncate=truncate,
        record=record,
        fps=fps,
        workers=workers
    )
    
    # Run
    trainer.train()