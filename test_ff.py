import math
import os
import sys
import time
from collections import deque

import torch
import torch.nn.functional as F
import torch.optim as optim
import tensorboard_logger as tb
import numpy as np
from gym import wrappers

from torch.autograd import Variable
from torchvision import datasets, transforms
from utils import logger
from utils.dblogger import DBLogger
from utils.misc import human_format

logger = logger.getLogger(__name__)

def test(rank, args, shared_model, Model, make_env, shared_stepcount):
    torch.manual_seed(args.seed + rank)
    
    dblogger = DBLogger(args.db_path)
    #import ipdb; ipdb.set_trace()
    env = make_env()
    env.seed(args.seed + rank)
    model = Model(env.observation_space.shape[0], env.action_space)
    model.eval()

    start_time = time.time()
    last_recorded_at = -math.inf  # global stepnumber of last recorded videos
    recording = False
    
    while True:
        global_step_count = shared_stepcount.get_value()
        # sync with the shared model
        model.load_state_dict(shared_model.state_dict())

        if recording:
            # we were recording in a previous cycle
            recording = False 
            env = make_env()
            #env.seed(args.seed + rank)

        if global_step_count-last_recorded_at > args.rec_every_nsteps:
            # recording time yeyy, create new env for now TODO check alternative
            env.close() # close current env
            env = make_env()
            #env.seed(args.seed + rank)
            rec_dir = os.path.join(args.checkpoint_dir, 'step_'+human_format(global_step_count))
            env = wrappers.Monitor(env, rec_dir, 
                    video_callable=lambda x: True, write_upon_reset=False)
            last_recorded_at = global_step_count
            recording = True
        
        episode_rewards = []
        episode_lengths = []
        for episode in range(args.num_test_episodes):
            state = torch.from_numpy(env.reset())
            episode_reward = 0
            episode_length = 1
            actions = deque(maxlen=100)
            while True: # episode is running
                value, logit = model(
                        Variable(state.unsqueeze(0), volatile=True))
                prob = F.softmax(logit)
                action = prob.max(1)[1].data.numpy()
                # a quick hack to prevent the agent from stucking
                actions.append(action[0, 0])
                state, reward, done, _ = env.step(action[0, 0])
                episode_reward += reward
                episode_length += 1
                state = torch.from_numpy(state)
                
                #if (done or actions.count(actions[0]) == actions.maxlen or 
                #        episode_length >= args.max_episode_length):
                # monitor shouldn't quit during recording
                if done:
                    break

            episode_rewards.append(episode_reward)
            episode_lengths.append(episode_length)
        
        env.close()
        # Do logging
        passed_time = time.time() - start_time
        avg_reward = np.average(episode_rewards)
        std_reward = np.std(episode_rewards)
        avg_length = np.average(episode_lengths)
        logger.info("GL step {}, Avg episode reward {}, avg episode length {}".format(
            global_step_count, avg_reward, avg_length))
        tb.log_value('steps_second', global_step_count / passed_time, global_step_count) 
        tb.log_value('reward', episode_rewards[0], global_step_count) 
        tb.log_value('avg_reward', avg_reward, global_step_count) 
        tb.log_value('std_reward', std_reward, global_step_count) 
        dblogger.info_reward(global_step_count, avg_reward, std_reward)
       
        time.sleep(60) # wait for a while
