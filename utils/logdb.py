# Very simple log keeper in sqlite database, where serialized data is stored
# snippet modified from github.com/scrapy/queuelib

import os
import sqlite3

class LogDB(object):
    ''' 
    keeps byte queue in an iterator style, next returns tuple of (id, bytes)
    '''

    _sql_tbl_create = '''
        CREATE TABLE log
        (id INTEGER PRIMARY KEY AUTOINCREMENT, 
        event CHARACTER(15),
        data BLOB,
        timestamp DATETIME DEFAULT CURRENT_TIMESTAMP
        )
    '''
    _sql_tbl_delete = 'DROP TABLE IF EXISTS log'
    _sql_size = 'SELECT COUNT(*) FROM log'
    _sql_push = 'INSERT INTO log (event, data) VALUES (?,?)'
    _sql_get = 'SELECT * FROM log Where id = ?'
    _sql_del = 'DELETE FROM log WHERE id = ?'
    _sql_max_id = 'SELECT * FROM log WHERE id=(SELECT MAX(id) FROM log)'
    _sql_exist_check =' SELECT 1 FROM log LIMIT 1'

    def __init__(self, path, role='consumer', overwrite=False):
        '''
           path : path to db file,
           role : 'consumer' or 'producer'
           overwrite : weather to overwrite a table if it exists
        '''
        self._path = os.path.abspath(path)
        self._db = sqlite3.Connection(self._path, timeout=60)
        self._db.text_factory = bytes
        with self._db as conn:
            try:
                conn.execute(self._sql_exist_check)
                exists = True
            except sqlite3.OperationalError:
                exists = False
            if role == 'producer':
                if not exists or overwrite is True:
                    conn.execute(self._sql_tbl_delete)
                    conn.execute(self._sql_tbl_create)
        self.last_fetched_id = 0
        self.role = role
    
    def __iter__(self):
        return self
    
    def max_id(self):
        with self._db as conn:
            max_item_id = conn.execute(self._sql_max_id).fetchone()
            if  max_item_id is None:
                return 0
            else:
                return max_item_id[0]

    def push(self, event, data):
        if self.role == 'consumer':
            raise ('push not suported for Consumer')

        if not isinstance(data, bytes):
            raise TypeError('Unsupported type: {}'.format(type(item).__name__))
        
        if not isinstance(event, str):
            raise TypeError('Unsupported type: {}'.format(type(event).__name__))

        with self._db as conn:
            conn.execute(self._sql_push, (event, data))

    def __next__(self):
        with self._db as conn:
            id_to_fetch = self.last_fetched_id + 1 
            if id_to_fetch > self.max_id():
                raise StopIteration
            record = conn.execute(self._sql_get, (id_to_fetch,))
            record = record.fetchone()
            if record is not None:
                self.last_fetched_id = id_to_fetch
                return record
            else:
                raise ('id {} was ommited'.format(id_to_fetch))

    def close(self):
        size = len(self)
        self._db.close()

    def __len__(self):
        with self._db as conn:
            return next(conn.execute(self._sql_size))[0]
