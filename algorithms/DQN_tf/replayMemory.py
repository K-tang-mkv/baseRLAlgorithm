# Store experience tuple:(current_image_state, action_selected, reward_got, next_image_observed) 
# into replay memory at each step the agent play. Sample a batch of tuples from replay memory 
# to train the neural network. 

from collections import deque, namedtuple   
import random 

Experience = namedtuple('experience', ('state', 'action', 'reward', 'next_state'))

class replayMemory(object):
    def __init__(self, capacity):
        self.experiences = deque([], capacity)

    def append(self, *args):
        self.experiences.append(Experience(*args))

    def sample(self, batch_size):
        return random.sample(self.experiences, batch_size)
    
    def __len__(self):
        return len(self.experiences)
