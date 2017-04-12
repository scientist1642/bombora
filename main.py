from __future__ import print_function

import argparse
import os
import sys
import math
import time
import logging
import inspect
import subprocess


import torch
import torch.optim as optim
import torch.multiprocessing as mp
import torch.nn as nn
import torch.nn.functional as F


from utils import logger
from utils import my_optim
from utils.shared_memory import SharedCounter

logger = logger.getLogger(__name__)

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
parser.add_argument('--num-processes', type=int, default=3, metavar='N',
                    help='how many training processes to use (default: 4)')
parser.add_argument('--num-steps', type=int, default=20, metavar='NS',
                    help='number of forward steps in A3C (default: 20)')
parser.add_argument('--max-episode-length', type=int, default=10000, metavar='M',
                    help='maximum length of an episode (default: 10000)')
parser.add_argument('--env-name', default='PongDeterministic-v3', metavar='ENV',
                    help='environment to train on (default: PongDeterministic-v3)')
parser.add_argument('--no-shared', default=False, metavar='O',
                    help='use an optimizer without shared momentum.')
parser.add_argument('--max-step-count', type=int, default=int(2e9),
                    help='maximum number of steps to run')
parser.add_argument('--debug', action='store_true', default=False,
                    help='run in a way its easier to debug')
parser.add_argument('--descr', default='nod',
                    help='Short description of the run params used for name')
parser.add_argument('--algo', default='a3c', dest='algo', action='store', choices=['a3c', 'a3cff'],
                    help='Algorithm to use')
parser.add_argument('--num-test-episodes', type=int, default=3, 
                    help='number of simple test episodes to run')
parser.add_argument('--test-simple-every', type=int, default=2,
                    help='intervals in minutes beteween simple test')
parser.add_argument('--test-heavy-every', type=int, default=30,
                    help='intervals in minutes beteween heavy test')
parser.add_argument('--log-source', action='store_true', default=True,
                    help='Log github hash commit of current code')

def setup_loggings(args, Model, train, test):
    ''' Setup python logging, tboard logging and dblogging '''
    logger.debug('CONFIGURATION: {}'.format(args))
    
    main_dir = os.path.dirname(os.path.realpath(__file__))
    run_stamp = '{}-{}'.format(args.descr, time.strftime('%d%m-%H%M'))
    run_title = run_stamp
    args.run_title = run_title
    run_dir = os.path.join(args.env_name, run_stamp)
    
    # dir for temp folders
    temp_dir = os.path.join(main_dir, 'dblogs', 'tempdir') 
    if not os.path.exists(temp_dir):
        os.makedirs(temp_dir)
    args.temp_dir = temp_dir

    #dir for sqlite log dbs
    db_dir = os.path.join(main_dir, 'dblogs', args.env_name)
    if not os.path.exists(db_dir):
        os.makedirs(db_dir)
    args.db_path = os.path.join(db_dir, run_title + '.sqlite3')
    
    logger.info('db log path is {}'.format(args.db_path))

    # Now let's keep the link to the current commit in args
    if args.log_source:
        hashcode = subprocess.check_output(['git', 'rev-parse', 'HEAD'])
        remote = subprocess.check_output(['git', 'remote', '-v'])
        hashcode, remote = map(lambda x: x.decode('utf-8').strip(), (hashcode, remote))
        remote_addr = remote.split()[1]
        if remote_addr.endswith('.git'):
            remote_addr = remote_addr[:-4]
        args.source_code = '{}/tree/{}'.format(remote_addr, hashcode)

def get_functions(args):
    ''' based on alg type return tuple of train/test functionsm model and env factory '''
    #TODO not very cool refactor later
    #TODO add make_model 
    if args.algo == 'a3c':
        from algorithms import a3c
        from models import actorcritic
        from test import test
        def make_env():
            from envs import create_atari_env42
            return create_atari_env42(args.env_name)
        return (a3c.train, test, actorcritic.ActorCritic, make_env)
    elif args.algo == 'a3cff':
        from algorithms import a3c_ff
        from models import nips_ff
        from test_ff import test
        def make_env():
            from envs import create_atari_env84
            return create_atari_env84(args.env_name)
        return (a3c_ff.train, test, nips_ff.ActorCritic, make_env)

    else:
        raise ('Unknown argument for algo')
        

if __name__ == '__main__':
    args = parser.parse_args()
    torch.manual_seed(args.seed)
    
    train, test, Model, make_env = get_functions(args) 
    setup_loggings(args, Model, train, test) 
    
    env = make_env()
    shared_model = Model(
        env.observation_space.shape[0], env.action_space)
    shared_model.share_memory()
    env.close()

    if args.no_shared:
        optimizer = None
    else:
        optimizer = my_optim.SharedAdam(shared_model.parameters(), lr=args.lr)
        optimizer.share_memory()
    
    shared_stepcount = SharedCounter()
    
    if not args.debug:
        processes = []

        p = mp.Process(target=test, args=(args.num_processes, args, 
            shared_model, Model, make_env, shared_stepcount))
        p.start()
        processes.append(p)
        for rank in range(0, args.num_processes):
            p = mp.Process(target=train, args=(rank, args, shared_model, 
                Model, make_env, shared_stepcount, optimizer))
            p.start()
            processes.append(p)
        for p in processes:
            p.join()
    else: ## debug is enabled
        # run only one process in a main, easier to debug
        args.max_step_count = 1000 # test both train and debug
        train(0, args, shared_model, Model, make_env, shared_stepcount, optimizer)
        args.max_step_count += 1000 # needed to perform test
        test(args.num_processes, args, shared_model, Model, make_env, shared_stepcount)

