import argparse

parser = argparse.ArgumentParser()
parser.add_argument('--env', type=str, help='Name of environment', default='HalfCheetah-v4')
parser.add_argument('--config', type=str, help='Path to json config')
parser.add_argument('--logdir', type=str, help='Path to log directory')
parser.add_argument('--steps', type=int, help='Total training steps', default=None)
parser.add_argument('--save_total', type=int, help='How many checkpoints', default=None)