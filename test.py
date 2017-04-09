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
        action_distr.append(logit.data.numpy()[0])
        
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
    
    #import ipdb; ipdb.set_trace()
    # do All the dblogging
    data = dblogging.TestHeavy(
            test_duration=time.time() - test_start_time,
            video=video_bytestr,
            states=np.stack(epstates),
            predvalues=np.stack(epvalues),
            action_distr=np.stack(action_distr),
            score=epreward,
            glsteps=glsteps,
            randomconv=np.stack(eprandomconv),
            )
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
    
    data = dblogging.TestSimple(
            glsteps=glsteps,
            avgscore=np.average(eprewards),
            stdscore=np.std(eprewards),
            avglength=np.average(eplengths),
            steps_second=glsteps / passed_time,
            )
    dblogger.log(data)
     


def test(rank, args, shared_model, Model, make_env, shared_stepcount):
    torch.manual_seed(args.seed + rank)
    dblogger = dblogging.DBLogging(args.db_path)
    
    # log experiment args
    dblogger.log(dblogging.ExperimentArgs(args=vars(args)))

    env = make_env()
    env.seed(args.seed + rank)
    model = Model(env.observation_space.shape[0], env.action_space)
    model.eval()
    start_time = time.time()
    
    testnum = 0
    #recording = False
    
    while True:
        testnum += 1
        glsteps = shared_stepcount.get_value()
        # sync with the shared model
        model.load_state_dict(shared_model.state_dict())
        
        if glsteps > args.max_step_count:
            # testing finished
            break
        test_simple(args, model, env, glsteps, dblogger, start_time)
        if testnum-1 % args.test_heavy_gap == 0:
            test_heavy(args, model, make_env, glsteps, dblogger)
        
        time.sleep(60) # wait for a while


'''
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
            rec_dir = tempfile.mkdtemp(dir=args.temp_dir)
            #rec_dir = tempfile.mkdtemp()
            #rec_dir = os.path.join(args.checkpoint_dir, 'step_'+human_format(global_step_count))
            env = wrappers.Monitor(env, rec_dir, 
                    video_callable=lambda x: True, write_upon_reset=False)
            last_recorded_at = global_step_count
            recording = True
        
        episode_rewards = []
        episode_lengths = []
        episode_videos = []
        episode_states = []
        episode_values = []
        for episode in range(args.num_test_episodes):
            state = torch.from_numpy(env.reset())
            cx = Variable(torch.zeros(1, 256), volatile=True)
            hx = Variable(torch.zeros(1, 256), volatile=True)
            
            episode_reward = 0
            episode_length = 1
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
                episode_reward += reward
                episode_length += 1
                if recording:
                    episode_states.append(state)
                    episode_values.append(value)

                state = torch.from_numpy(state)
                
                #if (done or actions.count(actions[0]) == actions.maxlen or 
                #        episode_length >= args.max_episode_length):
                # monitor shouldn't quit during recording
                if done:
                    break

            episode_rewards.append(episode_reward)
            episode_lengths.append(episode_length)
            if recording:
                # read recorded video 
                episode_videos.append(env.video_recorder.path)
       
        # remove created temp directory

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
        if recording:
            # read one of the file from recorded episodes
            with open(episode_videos[0], 'rb') as f:
                video_bytestr = f.read()
            dblogger.info_video(global_step_count, episode_rewards[0], video_bytestr)
            shutil.rmtree(rec_dir) #directory not needed
            dblogger.info_states(global_step_count, np.stack(episode_states))
            dblogger.info_pred_values(global_step_count, np.stack(episode_values))
       
        time.sleep(60) # wait for a while
'''
