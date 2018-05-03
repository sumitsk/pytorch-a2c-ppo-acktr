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
    if not os.path.exists(args.log_dir):
        os.makedirs(args.log_dir)

    agent = Agent(args)   
    if args.test:
        agent.test()
    else:
        agent.train()

if __name__ == "__main__":
    main()
