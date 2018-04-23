import glob
import os
import torch
import numpy as np

from arguments import get_args
from agent import Agent

def setup_seeds(use_cuda, seed):
    torch.manual_seed(seed)
    if use_cuda:
        torch.cuda.manual_seed(seed)
    np.random.seed(seed)



def main():
    args = get_args()
    setup_seeds(args.cuda, args.seed)

    if not os.path.exists(args.save_dir):
        os.makedirs(args.save_dir)
        
    try:
        os.makedirs(args.log_dir)
    except OSError:
        files = glob.glob(os.path.join(args.log_dir, '*.monitor.csv'))
        for f in files:
            os.remove(f)


    os.environ['OMP_NUM_THREADS'] = '1'

    agent = Agent(args)   
    if args.test:
        agent.test(args.model_filename)
    else:
        agent.train_maml()

if __name__ == "__main__":
    main()
