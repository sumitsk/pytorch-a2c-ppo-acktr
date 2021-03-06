import argparse


def get_args():
    parser = argparse.ArgumentParser(description='RL')
    parser.add_argument('--env-name',default='ant')
    parser.add_argument("--velocity-dir",default='posx',help='set one from: posx | posy | negx | negy ')
    parser.add_argument('--use-gym-obs',action='store_true',help='for using gym observation in rllab env')
    parser.add_argument('--use-gym-reward',action='store_true',help='for using gym reward in rllab env')

    parser.add_argument('--algo', default='ppo',help='algorithm to use: a2c | ppo | acktr')
    parser.add_argument('--ppo-epoch', type=int, default=10,help='number of ppo epochs (default: 10)')
    parser.add_argument('--num-mini-batch', type=int, default=32,help='number of batches for ppo (default: 32)')
    parser.add_argument('--clip-param', type=float, default=0.2,help='ppo clip parameter (default: 0.2)')
    parser.add_argument('--reward-scale',type=float,default=1.0,help='reward scaling factor')
    parser.add_argument('--penalty',type=float,default=3.0,help='penalty for velocity in perpendicular direction')
    parser.add_argument('--num-updates', type=int, default=int(10e6),help='number of updates to perform (default: 10e6)')
    parser.add_argument('--update-frequency', type=int, default=2,help='update model after every ... episodes')
    parser.add_argument('--episode-max-length', type=int,default=1000,help='terminate episode after ... steps')

    parser.add_argument('--lr', type=float, default=3e-4,help='learning rate (default: 3e-4)')    
    parser.add_argument('--gamma',type=float,default=0.995,help='discount factor for rewards (default: 0.99)')   
    parser.add_argument('--tau',type=float, default=0.98,help='gae parameter (default: 0.95)')
    parser.add_argument('--entropy-coef', type=float, default=0.01,help='entropy term coefficient (default: 0.01)')
    parser.add_argument('--value-loss-coef', type=float, default=0.5,help='value loss coefficient (default: 0.5)')
    parser.add_argument('--max-grad-norm', type=float, default=0.5,help='max norm of gradients (default: 0.5)')
    parser.add_argument('--model-filename',default='posx/AntEnv.pt')
    parser.add_argument('--test',action='store_true',help='use to test the model')
    parser.add_argument('--log-dir', default='./log_directory/',help='tensorboard log directory')
    parser.add_argument('--save-dir', default='./trained_models/',help='directory to save agent logs (default: ./trained_models/)')
    parser.add_argument('--no-gae', action='store_true', help='use generalized advantage estimation')
    parser.add_argument('--no-adam',action='store_true',help='use adam optimizer')
    parser.add_argument('--seed', type=int, default=1000,help='random seed (default: 10)')
    parser.add_argument('--cuda', action='store_true', help='disables CUDA training')
    parser.add_argument('--eps', type=float, default=1e-5,help='RMSprop optimizer epsilon (default: 1e-5)')
    parser.add_argument('--alpha',type=float,default=0.99,help='RMSprop optimizer apha (default: 0.99)')
    parser.add_argument('--save-interval', type=int, default=100,help='save interval, one save per ... updates (default: 100)')
    parser.add_argument('--eval-interval',type=int,default=25,help='evaluate model every ... updates')
    
    args = parser.parse_args()
    args.num_updates = int(args.num_updates)
    args.use_gae = not args.no_gae
    args.use_adam = not args.no_adam
    args.save_dir += args.velocity_dir + '/'
    return args
