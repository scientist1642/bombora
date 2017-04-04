import gym
import numpy as np
from gym.spaces.box import Box
import cv2

# Taken from https://github.com/openai/universe-starter-agent
def create_atari_env42(env_id):
    env = gym.make(env_id)
    box = Box(0.0, 1.0, [1, 42, 42])
    env = MyAtariRescale(env, _process_frame42, box)
    env = MyNormalizedEnv(env)
    return env

def create_atari_env84(env_id):
    env = gym.make(env_id)
    box = Box(0.0, 1.0, [4, 84, 84])
    env = MyAtariRescale(env, _process_frame84, box)
    env = MyNormalizedEnv(env)
    # also stack last 4 non skipped frames
    env = MyStackedFrames(env)
    return env

def _process_frame42(frame):
    frame = frame[34:34 + 160, :160]
    # Resize by half, then down to 42x42 (essentially mipmapping). If
    # we resize directly we lose pixels that, when mapped to 42x42,
    # aren't close enough to the pixel boundary.
    frame = cv2.resize(frame, (80, 80))
    frame = cv2.resize(frame, (42, 42))
    frame = frame.mean(2)
    frame = frame.astype(np.float32)
    frame *= (1.0 / 255.0)
    return frame

def _process_frame84(frame):
    frame = frame[34:34 + 160, :160]
    frame = cv2.resize(frame, (84, 84))
    frame = frame.mean(2)
    frame = frame.astype(np.float32)
    frame *= (1.0 / 255.0)
    return frame

class MyStackedFrames(gym.Wrapper):
    def __init__(self, env):
        super(MyStackedFrames, self).__init__(env)

    def _reset(self):
        observation = self.env.reset()
        self.stacked_observ = [observation] * 4
        return np.vstack(self.stacked_observ)

    def _step(self, action):
        observation, reward, done, info = self.env.step(action)
        self.stacked_observ = self.stacked_observ[1:] + [observation]
        return np.vstack(self.stacked_observ), reward, done, info


class MyAtariRescale(gym.ObservationWrapper):

    def __init__(self, env, pre_fun, box):
        super(MyAtariRescale, self).__init__(env)
        self.observation_space = box
        self.preprocess = pre_fun

    def _observation(self, observation):
        return self.preprocess(observation) 

class MyNormalizedEnv(gym.ObservationWrapper):

    def __init__(self, env=None):
        super(MyNormalizedEnv, self).__init__(env)
        self.state_mean = 0
        self.state_std = 0
        self.alpha = 0.9999
        self.num_steps = 0

    def _observation(self, observation):
        self.num_steps += 1
        self.state_mean = self.state_mean * self.alpha + \
            observation.mean() * (1 - self.alpha)
        self.state_std = self.state_std * self.alpha + \
            observation.std() * (1 - self.alpha)

        unbiased_mean = self.state_mean / (1 - pow(self.alpha, self.num_steps))
        unbiased_std = self.state_std / (1 - pow(self.alpha, self.num_steps))
        ret = (observation - unbiased_mean) / (unbiased_std + 1e-8)
        return np.expand_dims(ret, axis=0)
