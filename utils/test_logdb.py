import time
import pickle
from multiprocessing import Process
import numpy as np

from logdb import LogDB

def producer():
    pp = LogDB('temp', role='producer', overwrite=True)

    dict_item = {u'item1_dict': True, u'schema': 0}
    tuple_item = ('item2_tuple', 1)
    arr = np.arange(4).reshape(2,2)
    numpy_item = ['item3_numpyarray', arr]
    items = [dict_item, tuple_item, numpy_item]
    for item in items:
        pp.push('event_name', pickle.dumps(item))
        time.sleep(0.2)

def consumer():
    cc = LogDB('temp', role='consumer')
    for _ in range(5):
        for record in cc:
            idd, event, data, date = record
            unp_item = pickle.loads(data)
            print ('event {}, date  {}, data {}'.format(event, date, unp_item))
        time.sleep(0.5)

if __name__ == '__main__':
    ps = [Process(target=producer), Process(target=consumer)]
    ps[0].start()
    time.sleep(0.5)
    ps[1].start()
    for p in ps:
        p.join()
    
