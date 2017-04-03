import math
import os
import sys
import time

import torch
import torch.nn.functional as F
import torch.optim as optim
import tensorboard_logger as tb

from torch.autograd import Variable
from torchvision import datasets, transforms
from collections import deque
from utils import logger


logger = logger.getLogger(__name__)

def test(rank, args, shared_model, Model, make_env, shared_stepcount):
    torch.manual_seed(args.seed + rank)

    env = make_env()
    env.seed(args.seed + rank)

    model = Model(env.observation_space.shape[0], env.action_space)

    model.eval()

    state = env.reset()
    state = torch.from_numpy(state)
    reward_sum = 0
    done = True

    start_time = time.time()

    local_episode_num = 0 

    # a quick hack to prevent the agent from stucking
    actions = deque(maxlen=100)
    episode_length = 0
    while True:
        episode_length += 1
        # Sync with the shared model
        if done:
            model.load_state_dict(shared_model.state_dict())

        value, logit  = model(Variable(state.unsqueeze(0), volatile=True))
        prob = F.softmax(logit)
        action = prob.max(1)[1].data.numpy()

        state, reward, done, _ = env.step(action[0, 0])
        done = done or episode_length >= args.max_episode_length
        reward_sum += reward

        # a quick hack to prevent the agent from stucking
        actions.append(action[0, 0])
        if actions.count(actions[0]) == actions.maxlen:
            done = True

        if done:
            passed_time = time.time() - start_time
            local_episode_num += 1
            global_step_count = shared_stepcount.get_value()
            
            logger.info("Time {}, episode reward {}, episode length {}".format(
                time.strftime("%Hh %Mm %Ss",
                              time.gmtime(passed_time)),
                reward_sum, episode_length))
            tb.log_value('steps_second', global_step_count / passed_time, global_step_count) 
            tb.log_value('reward', reward_sum, global_step_count) 

            reward_sum = 0
            episode_length = 0
            actions.clear()
            state = env.reset()
            time.sleep(60)

        state = torch.from_numpy(state)
