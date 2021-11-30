from replayMemory import replayMemory
import observation
import gym
from dqn import DQN_agent
import matplotlib.pyplot as plt
from algorithms.DQN_pytorch_offical import plotting

if __name__ == "__main__":
    # Initialize replay memory D to capacity N
    replayMemory = replayMemory(10000)
    # Initialize action-value function Q with random weights
    
    env = gym.make('CartPole-v1')
    
    agent = DQN_agent(128, replayMemory, env)

    episodes_rewards = []
    # play 100 episodes
    for i in range(100):
        # at the begining of each episode, reset the env
        env.reset()
        current_state = observation.get_observation(env)
        print(i)
        total_reward = 0
        while True:
            action = agent.select_action(current_state)
            _, reward, done, _ = env.step(action)
            total_reward += reward
            next_state = observation.get_observation(env)

            experience = (current_state, action, reward, next_state)
            agent.replayMemory.append(current_state, action, reward, next_state)
            #print(len(replayMemory))
            agent.learn()
            if done:
                break
            else:
                current_state = next_state
        episodes_rewards.append(total_reward)
        plotting.plot_durations(episodes_rewards)
        print(i, " episode: ", total_reward)

    plt.show()

