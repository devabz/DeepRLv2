import json
import gymnasium as gym
from src import *


if __name__ == "__main__":
    args = parser.parse_args()
    with open(args.config, 'r') as f:
        config = json.load(f)
    
    
    if args.steps is not None:
        config['trainer']['total_training_steps'] = args.steps
    
    if args.save_total is not None:
        config['trainer']['save_total'] = args.save_total
        
    env = gym.make(args.env)
    trainer = build(config, env=env, logdir=args.logdir)
    trainer.train()