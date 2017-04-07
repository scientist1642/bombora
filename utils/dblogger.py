# main API to store things to db

'''
protocol
* VV - (VV, [rotocol version]) | init
* RW - (RW, step_number, reward_average, reward_std) | info_reward
* VD - (VD, step_number, [(run_reward1, video_path1) ...]) | info_video
'''
import pickle
from utils.logdb import LogDB

# TODO do validations

class DBLogger(object):
    def __init__(self, path):
        self.db = LogDB(path, role='producer')
        self._push(('VV', 1))
    
    def info_reward(self, step_number, reward_avg, reward_std):
        self._push(('RW', step_number, reward_avg, reward_std))
    
    def info_video(self, reward_video_tuples):
        ''' array of tuples (reward, video_path) '''
        self._push(('VD', reward_video_tuples))

    def _push(self, item):
        self.db.push(pickle.dumps(item))
