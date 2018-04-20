import argparse
import torch

def get_args():
    parser = argparse.ArgumentParser(description='RL')
    parser.add_argument(
            '--env-name',
            default='AntEnv')
    # HYPERPARAMETERS
    parser.add_argument(
            '--algo', 
            default='ppo',
            help='algorithm to use: a2c | ppo | acktr')
    # PPO hyperparameters
    parser.add_argument(
            '--ppo-epoch', 
            type=int, 
            default=10,
            help='number of ppo epochs (default: 10)')
    parser.add_argument(
            '--num-mini-batch', 
            type=int, 
            default=32,
            help='number of batches for ppo (default: 32)')
    parser.add_argument(
            '--clip-param', 
            type=float, 
            default=0.2,
            help='ppo clip parameter (default: 0.2)')

    parser.add_argument(
            '--num-updates', 
            type=int, 
            default=int(10e6),
            help='number of updates to perform (default: 10e6)')
    parser.add_argument(
            '--update-frequency', 
            type=int, 
            default=1,
            help='update model after every ... episodes')
    parser.add_argument(
            '--episode-max-length',
            type=int,
            default=1000,
            help='terminate episode after ... steps')

    # MOST SENSITIVE HYPERPARAMETERS
    parser.add_argument(
            '--lr', 
            type=float, 
            default=1e-4,
            help='learning rate (default: 7e-4)')    
    parser.add_argument(
            '--gamma',
            type=float,
            default=0.99,
            help='discount factor for rewards (default: 0.99)')   
    parser.add_argument(
            '--tau',
            type=float, 
            default=0.95,
            help='gae parameter (default: 0.95)')
    parser.add_argument(
            '--entropy-coef', 
            type=float, 
            default=0.01,
            help='entropy term coefficient (default: 0.01)')
    parser.add_argument(
            '--value-loss-coef', 
            type=float, 
            default=0.5,
            help='value loss coefficient (default: 0.5)')
    parser.add_argument(
            '--max-grad-norm', 
            type=float, 
            default=0.5,
            help='max norm of gradients (default: 0.5)')
    

    parser.add_argument(
            '--log-dir', 
            default='/tmp/gym/',
            help='directory to save agent logs (default: /tmp/gym)')
    parser.add_argument(
            '--save-dir', 
            default='./trained_models/',
            help='directory to save agent logs (default: ./trained_models/)')
    
    parser.add_argument(
            '--no-gae', 
            action='store_true', 
            help='use generalized advantage estimation')
    parser.add_argument(
            '--no-adam',
            action='store_true',
            help='use adam optimizer')
    parser.add_argument(
            '--seed', 
            type=int, 
            default=2018,
            help='random seed (default: 2018)')
    parser.add_argument(
            '--no-cuda', 
            action='store_true', 
            help='disables CUDA training')
    parser.add_argument(
            '--recurrent-policy', 
            action='store_true', 
            help='use a recurrent policy')
    parser.add_argument(
            '--no-vis', 
            action='store_true', 
            help='disables visdom visualization')
    parser.add_argument(
            '--port', 
            type=int, 
            default=8097,
            help='port to run the server on (default: 8097)')
    parser.add_argument(
            '--eps', 
            type=float, 
            default=1e-5,
            help='RMSprop optimizer epsilon (default: 1e-5)')
    parser.add_argument(
            '--alpha',
            type=float,
            default=0.99,
            help='RMSprop optimizer apha (default: 0.99)')
    parser.add_argument(
            '--log-interval', 
            type=int, 
            default=10,
            help='log interval, one log per ... updates (default: 10)')
    parser.add_argument(
            '--save-interval', 
            type=int, 
            default=100,
            help='save interval, one save per ... updates (default: 100)')
    parser.add_argument(
            '--vis-interval', 
            type=int, 
            default=100,
            help='vis interval, one log per ... updates (default: 100)')
    args = parser.parse_args()


    args.num_updates = int(args.num_updates)
    args.use_gae = not args.no_gae
    args.use_adam = not args.no_adam
    args.cuda = not args.no_cuda and torch.cuda.is_available()
    args.vis = not args.no_vis

    return args
