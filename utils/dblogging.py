# main API to store things to db

import pickle
from collections import namedtuple
from utils.logdb import LogDB
import zlib

TestSimple = namedtuple('TestSimple',[
        'glsteps',
        'avgscore',
        'avglength',
        'stdscore',
        'steps_second',   #steps trained per second
        ])

TestHeavy = namedtuple('TestHeavy',[
        'glsteps', # number of steps when the test was recorded
        'test_duration', # in seconds
        'video',   # bytestr video from the original game
        'states',  # n_xtep x state_shape,  states by the agent, preprocessed
        'action_distr', # n_step x num_act, containing probabilities
        'score',  # total score in the episode
        'predvalues',  # n_step x 1, predicted values for the state by network
        'randomconv', # n_step x conv_out_size x conv_out_size   activations of random channel in the first lauer
        ])

LoggerInfo = namedtuple('LoggerInfo', [
        'version', # db logger protocol version
        ]
)
ExperimentArgs = namedtuple('ExperimentArgs', [
        'args']
        )

class DBLogging:

    def __init__(self, path):
        self.db = LogDB(path, role='producer')
        self._push(LoggerInfo(version=1))
    
    def _push(self, data_tuple):
        # one of the defined named tuple which are defined above
        serialized = pickle.dumps(data_tuple)
        compressed = zlib.compress(serialized)
        self.db.push(compressed)

    def log(self, data_tuple):
        self._push(data_tuple)
    

class DBReader:
    # Iterator, simply wrapls LogDB 
    def __init__(self, path):
        self.db = LogDB(path, role='consumer')

    def __iter__(self):
        return self
    
    def __next__(self):
        idd, data = next(self.db)
        decompressed = zlib.decompress(data)
        unserialized = pickle.loads(decompressed)
        return unserialized
