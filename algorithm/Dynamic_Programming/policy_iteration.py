"""
Policy iteration is a dynamic programming algorithm based on the model.
The process of policy_iteration is divided by two parts, which are policy evaluation and policy improvement.
"""

from agent import Agent
import numpy as np


def policy_evaluation(agent, theta=0.01):
    while True:
        delta = 0.0  # stop condition
        for state in range(agent.nS):
            temp = 0.0
            for action in range(agent.nA):
                action_prob = agent.policy[state][action]
                temp += action_prob * agent.calc_action_value(state, action)  # calculate the expected
                # state value
            delta = max(delta, abs(temp - agent.values[state]))
            agent.values[state] = temp
        if delta < theta:
            break


def policy_improvement(agent):
    while True:
        policy_stable = True
        for state in range(agent.nS):
            temp = np.argmax(agent.policy[state])
            action = agent.select_action(state)  # it will greedy choose the best action
            agent.policy[state] = np.eye(agent.nA)[action]  # it keeps the prob of the best action 1 and elsewhere 0
            if temp != action:
                policy_stable = False
        if policy_stable:
            break
        else:
            policy_evaluation(agent)


def play_n_episode(agent, env, count=20):
    i = 0
    while i < count:
        i += 1
        print("Episode{0}:".format(i))
        policy_improvement(agent)
        agent.play_one_episode(env)


if __name__ == "__main__":
    import gym

    env = gym.make("FrozenLake-v0")
    agent = Agent(env)
    agent.play_n_random_steps(1000)  # produce the model(transition, reward)

    print(agent.transits)

    play_n_episode(agent, env, 10)
