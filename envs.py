import gym
import numpy as np
from gym.spaces.box import Box
import cv2

# Taken from https://github.com/openai/universe-starter-agent
def atari_env(env_id, side, stacked=1):
    ''' env_id: atari env id 
        side: square length to rescale, now either 42 or 84
        stacked: number of stacked frames
    '''
    if side not in [42, 84]:
        raise ('Unsuported frame size')
    env = gym.make(env_id)
    box = Box(0.0, 1.0, [stacked, side, side])
    env = MyAtariRescale(env, side, box)
    env = MyNormalizedEnv(env)
    if stacked > 1:
        env = MyStackedFrames(env, stacked)
    return env


class MyAtariRescale(gym.ObservationWrapper):

    def __init__(self, env, side, box):
        super(MyAtariRescale, self).__init__(env)
        self.observation_space = box
        self.side = side

    def _observation(self, observation):
        frame = observation
        frame = frame[34:34 + 160, :160]
        if self.side == 42:
            # Resize by half, then down to 42x42 (essentially mipmapping). If
            # we resize directly we lose pixels that, when mapped to 42x42,
            # aren't close enough to the pixel boundary.
            frame = cv2.resize(frame, (80, 80))
            frame = cv2.resize(frame, (42, 42))
        else:
            frame = cv2.resize(frame, (84, 84))
        frame = frame.mean(2)
        frame = frame.astype(np.float32)
        frame *= (1.0 / 255.0)
        return frame

class MyStackedFrames(gym.Wrapper):
    def __init__(self, env, stacked):
        super(MyStackedFrames, self).__init__(env)
        self.stacked = stacked

    def _reset(self):
        observation = self.env.reset()
        self.stacked_observ = [observation] * self.stacked
        return np.vstack(self.stacked_observ)

    def _step(self, action):
        observation, reward, done, info = self.env.step(action)
        self.stacked_observ = self.stacked_observ[1:] + [observation]
        return np.vstack(self.stacked_observ), reward, done, info

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



