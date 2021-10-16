import gym
"""
Enviorment spaces:
    1. observations space
    2. actions space
for the space class, we got the method sample(), which can select samples(observations or actions) from the two spaces randomly.

Test:
    reset() function, # return the original observation
    sample() function,
    step() function
"""
if __name__ == "__main__":
    env = gym.make("CartPole-v0")
    obs = env.reset() # test reset() function, which will return the list of observations
    print("Initial observations: ", obs)
    # in the cartPole env, observations = [x_coordinate of the stick, speed, stick angle, angular speed]
    
    obs_sample = env.observation_space.sample() # test the sample()
    actions = env.action_space.sample()
    # two actions in this env, [0(left), 1(right)]
    print("observation_sample: {0} \naction_sample: {1}".format( obs_sample, actions))
    print("the number of actions: ", env.action_space.n)
    obs, reward, done, _ = env.step(actions)
    # test the step() function, which accept an action as parameter, actually the agent will perform the action, and then get the observation, reward, done_flag, and the information of the env.
    print("obs: {0} \nreward: {1} \ndone: {2} \n_: {3}".format(obs, reward, done, _))
