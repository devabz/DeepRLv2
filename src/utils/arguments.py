import argparse

parser = argparse.ArgumentParser()
parser.add_argument('--config', type=str, help='Path to json config')
parser.add_argument('--logdir', type=str, help='Path to log directory')
parser.add_argument('--steps', type=int, help='Total training steps', default=int(1e6))