class ReplayBuffer():

    def __init__(self):
        # TODO
        pass

    def add(self,state, action, reward, next_state, done):
        """ Add new experience to the buffer"""
        pass
    # TODO

    def get_batch(self, batch_size=32):
        pass
    # TODO

    def save(self, folder_name):
        pass

    def load(self, folder_name):
        pass

class PrioritzedBuffer():
    # TODO: Nochmal schauen, welche Methoden wir hier noch brauchen
    def __init__(self):
        super(ReplayBuffer,self).__init__()
        pass
    # TODO

