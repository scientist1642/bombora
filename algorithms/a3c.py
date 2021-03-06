import math
import os
import sys

import torch
import torch.nn.functional as F
import torch.optim as optim


from torch.autograd import Variable
from torchvision import datasets, transforms
from utils import logger

logger = logger.getLogger(__name__)

def ensure_shared_grads(model, shared_model):
    for param, shared_param in zip(model.parameters(), shared_model.parameters()):
        if shared_param.grad is not None:
            return
        shared_param._grad = param.grad

def train(rank, args, shared_model, Model, make_env, gl_step_count, optimizer=None):
    torch.manual_seed(args.seed + rank)

    env = make_env()
    env.seed(args.seed + rank)

    model = Model(args.input_shape, args.num_actions)

    if optimizer is None:
        optimizer = optim.Adam(shared_model.parameters(), lr=args.lr)

    model.train()

    state = env.reset()
    state = torch.from_numpy(state)
    done = True

    episode_length = 0
    episode_count = 0 
    
    while True:

        values = []
        log_probs = []
        rewards = []
        entropies = []
        
        if gl_step_count.get_value() >= args.max_step_count:
            logger.info('Maxiumum step count {} reached..'.format(args.max_step_count))
            # TODO make sure if no train process is running test.py closes  as well
            break

        episode_length += 1

        # Sync with the shared model
        model.load_state_dict(shared_model.state_dict())
        if done:
            cx = Variable(torch.zeros(1, 256))
            hx = Variable(torch.zeros(1, 256))
        else:
            cx = Variable(cx.data)
            hx = Variable(hx.data)


        for step in range(args.num_steps):
            value, logit, (hx, cx) = model(
                (Variable(state.unsqueeze(0)), (hx, cx)))
            prob = F.softmax(logit)
            log_prob = F.log_softmax(logit)
            entropy = -(log_prob * prob).sum(1)
            entropies.append(entropy)
            action = prob.multinomial().data
            log_prob = log_prob.gather(1, Variable(action))
            state, reward, done, _ = env.step(action[0,0])
            done = done or episode_length >= args.max_episode_length
            reward = max(min(reward, 1), -1)

            if done:
                episode_length = 0
                episode_count += 1
                state = env.reset()

            state = torch.from_numpy(state)
            values.append(value)
            log_probs.append(log_prob)
            rewards.append(reward)

            if done:
                break

        # increment global step count
        gl_step_count.increment_by(step)

        R = torch.zeros(1, 1)
        if not done:
            value, _, _ = model((Variable(state.unsqueeze(0)), (hx, cx)))
            R = value.data

        values.append(Variable(R))
        policy_loss = 0
        value_loss = 0
        R = Variable(R)
        gae = torch.zeros(1, 1)
        for i in reversed(range(len(rewards))):
            R = args.gamma * R + rewards[i]
            advantage = R - values[i]
            value_loss = value_loss + 0.5 * advantage.pow(2)

            # Generalized Advantage Estimataion
            delta_t = rewards[i] + args.gamma * \
                values[i + 1].data - values[i].data
            gae = gae * args.gamma * args.tau + delta_t

            policy_loss = policy_loss - \
                log_probs[i] * Variable(gae) - 0.01 * entropies[i]

        optimizer.zero_grad()

        (policy_loss + 0.5 * value_loss).backward()
        torch.nn.utils.clip_grad_norm(model.parameters(), 40)

        ensure_shared_grads(model, shared_model)
        optimizer.step()
