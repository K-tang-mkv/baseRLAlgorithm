# Replay Memory
# In order to reduce the corelation between two continous images so that it will improve a lot performance 
# of the DNN. At the same time, the whole neuron network will be more stablizing by the training process. 
# We can select a batch of samples from the replay memory randomly. It stores transitions that the agent 
# observes. 
# 

from collections import namedtupe
import random   

# function namedtupe will return a subclass of tuple, namedtupe, which is the same as tuple with the name
Transition = namedtupe('Transition', ('state', 'action', 'next_state', 'reward'))

# Another class we need is the replay memory class which stores transitions and has a method .sample() which
# can be used for us to select a sample (transition) from the replay memory randomly
class ReplayMemory(object):
    def __init__(self, transitions):
        self.transitions = transitions 
    
    def putMemory(self, transition):
        self.transitions.append(transition)
    
    def sample(self, batch_size):
        # return a number of transitions with the batchSize 
        
        return random.sample(self.transitions, batch_size)

    # return the size of the memory
    def __len__(self):
        return len(self.transitions)






