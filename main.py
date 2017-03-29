from __future__ import print_function

import argparse
import os
import sys
import math
import time

import torch
import torch.optim as optim
import torch.multiprocessing as mp
import torch.nn as nn
import torch.nn.functional as F
import tensorboard_logger as tb

import my_optim
from envs import create_atari_env
from model import ActorCritic
from train import train
from test import test
from utils import logger
from utils.shared_memory import SharedCounter


logger = logger.getLogger('main')

# Based on
# https://github.com/pytorch/examples/tree/master/mnist_hogwild
# Training settings
parser = argparse.ArgumentParser(description='A3C')
parser.add_argument('--lr', type=float, default=0.0001, metavar='LR',
                    help='learning rate (default: 0.0001)')
parser.add_argument('--gamma', type=float, default=0.99, metavar='G',
                    help='discount factor for rewards (default: 0.99)')
parser.add_argument('--tau', type=float, default=1.00, metavar='T',
                    help='parameter for GAE (default: 1.00)')
parser.add_argument('--seed', type=int, default=1, metavar='S',
                    help='random seed (default: 1)')
parser.add_argument('--num-processes', type=int, default=4, metavar='N',
                    help='how many training processes to use (default: 4)')
parser.add_argument('--num-steps', type=int, default=20, metavar='NS',
                    help='number of forward steps in A3C (default: 20)')
parser.add_argument('--max-episode-length', type=int, default=10000, metavar='M',
                    help='maximum length of an episode (default: 10000)')
parser.add_argument('--env-name', default='PongDeterministic-v3', metavar='ENV',
                    help='environment to train on (default: PongDeterministic-v3)')
parser.add_argument('--no-shared', default=False, metavar='O',
                    help='use an optimizer without shared momentum.')
parser.add_argument('--max-episode-count', type=int, default=math.inf,
                    help='maximum number of episodes to run per process.')
parser.add_argument('--debug', action='store_true', default=False,
                    help='run in a way its easier to debug')
parser.add_argument('--short-description', default='no_descr',
                    help='Short description of the run params, (used in tensorboard)')

def setup_loggings(args):
    logger.debug('CONFIGURATION: {}'.format(args))
    
    cur_path = os.path.dirname(os.path.realpath(__file__))
    args.summ_base_dir = (cur_path+'/runs/{}/{}({})').format(args.env_name, 
            time.strftime('%d.%m-%H.%M'), args.short_description)
    logger.info('logging run logs to {}'.format(args.summ_base_dir))
    tb.configure(args.summ_base_dir)

if __name__ == '__main__':
    args = parser.parse_args()
    setup_loggings(args) 
    torch.manual_seed(args.seed)

    env = create_atari_env(args.env_name)
    shared_model = ActorCritic(
        env.observation_space.shape[0], env.action_space)
    shared_model.share_memory()

    if args.no_shared:
        optimizer = None
    else:
        optimizer = my_optim.SharedAdam(shared_model.parameters(), lr=args.lr)
        optimizer.share_memory()
    
    gl_step_cnt = SharedCounter()
    
    if not args.debug:
        processes = []

        p = mp.Process(target=test, args=(args.num_processes, args, 
            shared_model, gl_step_cnt))
        p.start()
        processes.append(p)
        for rank in range(0, args.num_processes):
            p = mp.Process(target=train, args=(rank, args, shared_model, 
                gl_step_cnt, optimizer))
            p.start()
            processes.append(p)
        for p in processes:
            p.join()
    else: ## debug is enabled
        # run only one process in a main, easier to debug
        train(0, args, shared_model, gl_step_cnt, optimizer)
