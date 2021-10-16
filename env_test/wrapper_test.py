import gym
import random

"""
Wrapper is designed for us to costmize the extensive methods in the env
classes. Wrapper is actually inherited by these env. There are three
basic inheritance class of Wrapper, which are:
    ~ObservationWrappers: Redefine the observation method of the parent.
    The obs argument is an observation passed to the wrapped envs to
    the agent.
    
    ~RewardWrapper: Modify the reward value given to the agent.
    
    ~ActonWrapper: Tweak the action passed to the wrapped envs to the agent.
"""
"""
built-in Wrapper class:

class ObservationWrapper(Wrapper):
    def reset(self, **kwargs):
        observation = self.env.reset(**kwargs)
        return self.observation(observation)

    def step(self, action):
        observation, reward, done, info = self.env.step(action)
        return self.observation(observation), reward, done, info

    def observation(self, observation):
        raise NotImplementedError


class RewardWrapper(Wrapper):
    def reset(self, **kwargs):
        return self.env.reset(**kwargs)

    def step(self, action):
        observation, reward, done, info = self.env.step(action)
        return observation, self.reward(reward), done, info

    def reward(self, reward):
        raise NotImplementedError

class ActionWrapper(Wrapper):
    def reset(self, **kwargs):
        return self.env.reset(**kwargs)

    def step(self, action):
        return self.env.step(self.action(action))

    def action(self, action):
        raise NotImplementedError

    def reverse_action(self, action):
        raise NotImplementedError
"""

class ObsWrapper(gym.ObservationWrappers):
    def __init__(self, env):
        super().__init__(env)
    
    def observation(self, obs):
        pass
"""
class IncreaseRewardWrapper(RewardWrapper):
    def __init__(self, env):
        super().__init__(env)
    
    def reward(self, reward):
        if random.random() < 0.3:
            return 2
"""
class RandomActionWrapper(gym.ActionWrapper):
    def __init__(self, env, epsilon=0.1):
        super(RandomActionWrapper, self).__init__(env) # call the super class's __init__()
        self.epsilon = epsilon
    
    def action(self, action):
        if random.random() < self.epsilon:
            print("Random!")
            return self.env.action_space.sample()
        return action

if __name__ == "__main__":
    env = RandomActionWrapper(gym.make("CartPole-v0"))
    obs = env.reset() # call the original class's method
    total_reward = 0.0
    
    while True:
        obs, reward, done, _ = env.step(0)
        total_reward += reward
        if done:
            break
    print("Reward get: %.2f" % total_reward)
