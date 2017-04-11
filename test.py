import math
import os
import sys
import time
import tempfile
import shutil
from collections import deque

import torch
import torch.nn.functional as F
import torch.optim as optim
import tensorboard_logger as tb
import numpy as np
from gym import wrappers

from utils import dblogging
from torch.autograd import Variable
from torchvision import datasets, transforms
from utils import logger
from utils.misc import human_format

logger = logger.getLogger(__name__)

def test_heavy(args, model, make_env,  glsteps, dblogger):
    logger.info('Doing heavy test on step {}'.format(glsteps))
    test_start_time = time.time()
    env = make_env() # should be closed
    recdir = tempfile.mkdtemp(dir=args.temp_dir) # should be closed
    
    env = wrappers.Monitor(env, recdir, 
            video_callable=lambda x: True, write_upon_reset=False)

    state = torch.from_numpy(env.reset())
    cx = Variable(torch.zeros(1, 256), volatile=True)
    hx = Variable(torch.zeros(1, 256), volatile=True)
     
    epreward = 0
    eplength = 1
    epstates = []
    epvalues = [] # predicted values
    action_distr = []
    eprandomconv=[]
    actions = deque(maxlen=100)
    while True: # episode is running
        inputs = (Variable(state.unsqueeze(0), volatile=True), (hx, cx))
        value, logit, (hx, cx, conv1_out) = model(inputs, req_params=['conv1_out'])
        cx = Variable(cx.data, volatile=True)
        hx = Variable(hx.data, volatile=True)
        
        prob = F.softmax(logit)
        action = prob.max(1)[1].data.numpy()
        
        eprandomconv.append(conv1_out.data.numpy()[0][11])   # save random conv activations for analyze # unbiased random num is 11 :)
        actions.append(action[0, 0])
        state, reward, done, _ = env.step(action[0, 0])
        epreward += reward
        eplength += 1
        epstates.append(state)
        epvalues.append(value.data.numpy()[0,0])
        action_distr.append(prob.data.numpy()[0])
        
        state = torch.from_numpy(state)
        
        #if (done or actions.count(actions[0]) == actions.maxlen or 
        #        episode_length >= args.max_episode_length):
        # monitor shouldn't quit during recording
        if done:
            break

    env.close()
    with open(env.video_recorder.path, 'rb') as f:
        video_bytestr = f.read()
    
    shutil.rmtree(recdir) 
    
    # do All the dblogging
    data = {'evtname':'HeavyTest',
            'test_duration':time.time() - test_start_time,
            'video':video_bytestr,
            'states':np.stack(epstates),
            'predvalues':np.stack(epvalues),
            'action_distr':np.stack(action_distr),
            'score':epreward,
            'glsteps':glsteps,
            'randomconv':np.stack(eprandomconv),
            }
    dblogger.log(data)
    logger.info('Finished heavy test on step {}'.format(glsteps))

def test_simple(args, model, env, glsteps, dblogger, start_time):
    logger.info('Doing simple test on step {}'.format(glsteps))
    eprewards = []
    eplengths = []
    env.reset()
    for episode in range(args.num_test_episodes):
        state = torch.from_numpy(env.reset())
        cx = Variable(torch.zeros(1, 256), volatile=True)
        hx = Variable(torch.zeros(1, 256), volatile=True)
        
        epreward = 0
        eplength = 1
        actions = deque(maxlen=100)
        while True: # episode is running
            value, logit, (hx, cx) = model(
                    (Variable(state.unsqueeze(0), volatile=True), (hx, cx)))

            cx = Variable(cx.data, volatile=True)
            hx = Variable(hx.data, volatile=True)

            prob = F.softmax(logit)
            action = prob.max(1)[1].data.numpy()
            # a quick hack to prevent the agent from stucking
            actions.append(action[0, 0])
            state, reward, done, _ = env.step(action[0, 0])
            epreward += reward
            eplength += 1

            state = torch.from_numpy(state)
            
            #if (done or actions.count(actions[0]) == actions.maxlen or 
            #        episode_length >= args.max_episode_length):
            # monitor shouldn't quit during recording
            if done:
                break

        eprewards.append(epreward)
        eplengths.append(eplength)
    
    env.close()


    # Do Db loggings
    passed_time = time.time() - start_time

    data = {'evtname':'SimpleTest',
            'glsteps': glsteps,
            'avgscore': np.average(eprewards),
            'stdscore': np.std(eprewards),
            'avglength': np.average(eplengths),
            'tpassed' : passed_time,
            }

    dblogger.log(data)

    logger.info('Finished simple test on step {}'.format(glsteps))

def test(rank, args, shared_model, Model, make_env, shared_stepcount):
    torch.manual_seed(args.seed + rank)
    dblogger = dblogging.DBLogging(args.db_path)
    
    # log experiment args
    dblogger.log({'evtname':'ExperimentArgs', 'args': vars(args)})

    env = make_env()
    env.seed(args.seed + rank)
    model = Model(env.observation_space.shape[0], env.action_space)
    model.eval()
    start_time = time.time()
    
    testnum = 0
    #recording = False
    
    while True:
        glsteps = shared_stepcount.get_value()
        # sync with the shared model
        model.load_state_dict(shared_model.state_dict())
        
        if glsteps > args.max_step_count:
            # testing finished
            break
        if testnum  % args.test_simple_every == 0:
            test_simple(args, model, env, glsteps, dblogger, start_time)
        
        if testnum % args.test_heavy_every == 0:
            test_heavy(args, model, make_env, glsteps, dblogger)
        
        testnum += 1
        time.sleep(60) # wait for a minute
