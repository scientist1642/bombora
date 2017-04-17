# LOAD AND STORE DESIRED MODEL CHECKPOINT AS A FILE 
# TODO refactor
import sys
import io
import tempfile
import numpy as np
import torch
from torch.autograd import Variable
from utils.dblogging import DBReader

if len(sys.argv) == 1:
    print ('Please pass db path')
else:
    path = sys.argv[1]

db = DBReader(path)

count = 0
for idd, evtname, data, timestamp in db:
    if data['evtname'] == 'ModelCheckpoint':
        count += 1
        latest_params = data['state_dict']
        # Just a test
        if data['algo'] == 'a3c' and data['arch'] == 'lstm_universe':
            from models.lstm_universe import Net
            model = Net(data['num_channels'], data['num_actions'])
            with tempfile.NamedTemporaryFile() as f:
                f.write(data['state_dict'])
                print ('param size in mb {}'.format(len(data['state_dict'])/1024/1024))
                state_dict = torch.load(f.name)
                model.load_state_dict(state_dict)
                cx = Variable(torch.zeros(1, 256), volatile=True)
                hx = Variable(torch.zeros(1, 256), volatile=True)
                state = torch.from_numpy(np.random.rand(1,1,42,42).astype('float32'))
                value, logit, (hx, cx) = model((Variable(state), (hx, cx)))
                print (logit)
                print ("Everything cool!")

print ('encountered {} model params'.format(count))

if count > 0:
    if len(sys.argv) == 2:
        print ('to write, please provide file name as a second argument')
    else:
        outname = sys.argv[2]
        with open(outname, 'wb') as f:
            f.write(latest_params)
            print ('Wrote params to {}'.format(outname))

