# Store experience tuple:(current_image_state, action_selected, reward_got, next_image_observed) 
# into replay memory at each step the agent play. Sample a batch of tuples from replay memory 
# to train the neural network. 

from collections import deque        
import random 
class replayMemory(object):
    def __init__(self, capacity):
        self.experiences = deque([], capacity)

    def store(self, experience):
        self.experiences.append(experience)

    def sample(self):
        random.sample(self.experiences, len(self.experiences)/10)
    
    def __len__(self):
        return len(self.experiences)
