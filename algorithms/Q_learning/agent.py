# agent with q_learning
import random
import numpy as np
import pandas as pd

class QLAgent(object):
    def __init__(self, env, epsilon=1, discount_factor=0.9):
        self.env = env
        self.epsilon = epsilon
        self.gamma = discount_factor
        self.action_space = env.action_space # action_space: [0,1,2,3] -> [up, down, left, right]
        self.actions = [i for i in range(len(self.action_space))]

        self.qvalue_table = pd.DataFrame(columns=self.action_space)


    def select_action(self, observation):
        """
            based on epsilon-greedy policy to select an action
        """
        # check if this observation exists
        self.check_observation(observation)
        alpha = random.random()

        if self.epsilon < alpha:
            if (self.qvalue_table.loc[observation][0] == self.qvalue_table.loc[observation]).all():
                return random.choice(self.actions) # if there are the same values
            return np.argmax(self.qvalue_table.loc[observation])  # return the index of the maximum of the actions over observation
        else:
            return random.choice(self.actions) # return a random action


    def learn(self, current_obs, action, next_obs, reward):
        self.check_observation(next_obs)
        self.qvalue_table.loc[current_obs] = reward + self.gamma * self.qvalue_table.loc[next_obs].max()
        self.epsilon -= 0.001


    def check_observation(self, observation):
        if observation in self.qvalue_table.index:
            return True
        else:
            self.qvalue_table = self.qvalue_table.append(pd.Series(
                data=[0]*len(self.action_space),
                index=self.qvalue_table.columns,
                name=observation
            ))
            return False
