# main API to store things to db

import pickle
from collections import namedtuple
from utils.logdb import LogDB
import zlib
from voluptuous import Schema
from voluptuous import Or
import numpy as np

# name of the event, data should be python seriazible dict

num = Or(float, int)

evttypes = {
        'LoggerInfo': Schema({
            'evtname':'LoggerInfo',
            'version': int,
            }, required=True),
        
        'SimpleTest': Schema({
            'evtname': 'SimpleTest', 
            'glsteps': int,
            'avgscore': float,
            'avglength': float,
            'stdscore': float,
            'tpassed': float
            }, required=True),
        
        'HeavyTest': Schema({
            'evtname':'HeavyTest',
            'glsteps': int, # number of steps when the test was recorded
            'test_duration': float, # in second
            'video':bytes,   # bytestr video from the original game
            'states': np.ndarray, # n_xtep x state_shape,  states by the agent, preprocessed
            'action_distr': np.ndarray, # n_step x num_act, containing probabilities
            'score':  num, # total score in the episode
            'predvalues':   np.ndarray, # n_step x 1, predicted values for the state by network
            'randomconv': np.ndarray, # n_step x conv_out_size x conv_out_size   activations of random channel in the first lauer
            'actions': list, #chosen actions at each timestap
            }, required=True),
        
        'ExperimentArgs': Schema({
            'evtname':'ExperimentArgs',
            'args': dict,   # argparse arg
            'action_names': list,  # action descriptions
            }, required=True),

        'ModelCheckpoint': Schema({
            'evtname':'ModelCheckpoint',
            'glsteps': int,   
            'algo': str,
            'arch': str, # architechture
            'tpassed': float, 
            'num_channels': int, # first dim of observation
            'num_actions': int,  # number of actions in environment
            'state_dict':bytes, # serialized model.state_dict() 
            }, required=True)
}

class DBLogging:

    def __init__(self, path):
        self.db = LogDB(path, role='producer')
        VERSION=2
        self.log({'evtname':'LoggerInfo', 'version':1})
    
    def _push(self, name, data):
        serialized = pickle.dumps(data)
        compressed = zlib.compress(serialized)
        self.db.push(name, compressed)

    def log(self, data):
        evtname = data['evtname']
        if evtname not in evttypes:
            raise TypeError('Unknown event type')
        
        sch = evttypes[evtname]
        data = sch(data)
        self._push(evtname, data)

class DBReader:
    # Iterator, simply wrapls LogDB 
    def __init__(self, path):
        self.db = LogDB(path, role='consumer')

    def __iter__(self):
        return self
    
    def __next__(self):
        idd, evtname, data, timestamp = next(self.db)
        decompressed = zlib.decompress(data)
        unserialized = pickle.loads(decompressed)
        return (idd, evtname, unserialized, timestamp)
