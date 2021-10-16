"""

For the discrete reinforecement learning, we usually do some tabulars
updation. There are four tables with pandas data structure type in this part:
    ~ Reward table: A two dimensional array Pd.DataFrame with the composite key "current state" + "action". The column index is "next_state". The value is obtained from the immediate
    reward. just like:
        [state, action][next_state]=reward} # it means performing action with given state and move to next state and get a reward

    ~ Transitions table: A table is like the reward table, but the value is the probability of moving from one state with given action to another state. like: [s,a][next_s]=0.'''

    ~ Value table: A one dimensional array Pd.series with the rows index states and the cols index value. the length of the array is the number of states

    ~ Policy table: A two dimensional array Pd.DataFrame with the rows index states and the cols index action, and the value is the probability of performing one action given one state.

"""

import numpy as np
import pandas as pd
import collections
import time


class Agent(object):
    """
    Construtor for us to sample data, obtain our first observation,
    and define three tables.
    """

    def __init__(self, env):
        self.env = env
        self.state = self.env.reset()
        self.nA = self.env.action_space.n  # the number of actions
        self.nS = self.env.observation_space.n  # the number of states

        def create_col():
            col = []
            for i in range(self.nS):
                for j in range(self.nA):
                    col.append((i, j))
            return col

        # three tables
        self.rewards = collections.defaultdict(float)
        self.transits = pd.DataFrame(columns=create_col(), index=range(self.nS), dtype=float).fillna(0.0)
        self.values = np.random.rand(self.nS)
        self.policy = np.ones((self.nS, self.nA)) / self.nA

    """
    Play some random steps from the environment, populating the reward
    and transition tables (model).
    """

    def play_n_random_steps(self, count):
        for i in range(count):
            action = self.env.action_space.sample()
            next_state, reward, is_done, _ = self.env.step(action)

            # update two tables
            self.rewards[(self.state, action, next_state)] = reward
            # the count of times of the transition between two states increase one time
            self.transits[(self.state, action)][next_state] += 1

            if is_done:
                self.state = self.env.reset()  # next episode
            else:
                self.state = next_state

    def cal_trans_prob(self, state, action, next_state):
        """
        This function will calculate the probability P(s'|(s, a)).
        """
        if self.transits[(state, action)].sum() == 0.0:
            prob = 0.0
        else:
            prob = self.transits[(state, action)][next_state] / self.transits[(state, action)].sum()
        return prob

    """
    The next function is to calculate the value of action from the sta-
    te, using our transition, reward and values tables. Q-value: Q(s,a)
    """

    def calc_action_value(self, state, action, gamma=0.9):
        q_value = 0.0
        for next_state in self.transits[(state, action)].index:
            reward = self.rewards[(state, action, next_state)]
            # calculate the probability of transition
            prob = self.cal_trans_prob(state, action, next_state)
            q_value += prob * (reward + gamma * self.values[next_state])
        return q_value

    """
    Use greedy policy to select an action given the state based on the
    actions value
    """

    def select_action(self, state):
        best_action, best_value = None, None
        for action in range(self.env.action_space.n):
            action_value = self.calc_action_value(state, action)
            if best_value is None or best_value < action_value:
                best_value = action_value
                best_action = action
        return best_action  # return the action with Max(Q-value)

    def play_one_episode(self, env):
        state = env.reset()
        env.render()
        total_reward = 0
        iter_num = 0
        while True:
            action = self.policy[state].argmax()
            next_state, reward, is_done, _ = env.step(action)
            # update two tables
            self.rewards[(self.state, action, next_state)] = reward
            # the count of times of the transition between two states increase one time
            self.transits[(self.state, action)][next_state] += 1
            env.render()
            time.sleep(0.1)
            total_reward += reward
            iter_num += 1
            if is_done:
                break
            else:
                state = next_state

        env.close()
        print("total_reward: {0}".format(total_reward))
        print("step_num: ", iter_num)
