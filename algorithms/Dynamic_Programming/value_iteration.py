"""
Value iteration algorithm is for the discrete Markov-decsion process b-
ased on model, which means that the reward and the transition probabil-
ity are known.
Value iteration algorithm allows to numerically calculate the values of
states and values of actions of MDPs with known transition probabiliti-
es and rewards. The procedure is like below:
    1. Initialize values of all states Vi to some initial value(zero)
    2. For every state s in the MDP, perform the Bellman update to calculate Q-value, and then V(s) = max(Q-value);
    3. Repeated step 2 for some large number of steps or until changes
    become too small.

"""
from agent import Agent
import numpy as np


# the detail of this algorithm referenced https://i.stack.imgur.com/wGuj5.png
def value_iteration(agent, env, theta=0.001):
    while True:
        delta = 0  # stopping condition
        # update each state value
        for state in range(agent.nS):
            temp = agent.values[state]
            state_values = [agent.calc_action_value(state, action) for action in range(env.action_space.n)]
            agent.values[state] = max(state_values)  # using greedy selection
            delta = max(0, abs(temp - agent.values[state]))

        if delta < theta:
            break

    # Create a deterministic policy using the optimal value function
    for state in range(agent.nS):
        action = agent.select_action(state)  # this will return the best action
        agent.policy[state] = np.eye(agent.nA)[action]

    return agent.policy


def play_n_episode(agent, env, count=20):
    i = 0
    while i < count:
        i += 1
        print("Episode{0}:".format(i))
        value_iteration(agent, env)
        agent.play_one_episode(env)


if __name__ == "__main__":
    import gym

    env = gym.make("FrozenLake-v0")
    agent = Agent(env)
    agent.play_n_random_steps(1000)  # produce the model(transition, reward)

    print(agent.transits)

    play_n_episode(agent, env, 10)
