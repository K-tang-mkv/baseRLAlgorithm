import gym 
import torch

import matplotlib
from matplotlib import pyplot as plt


env = gym.make("CartPole-v1").unwrapped

# set up matplotlib
is_ipython = 'inline' in matplotlib.get_backend()
if is_ipython:
    from IPython import display     

plt.ion()

# if gpu is to be used
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
