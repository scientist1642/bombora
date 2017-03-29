from multiprocessing import Value, Lock

class SharedCounter:
    def __init__(self):
        self.lock = Lock()
        self.n = Value('i', 0)

    def increment_by(self, k):
        with self.lock:
            self.n.value += k

    def get_value(self):
        return self.n.value
