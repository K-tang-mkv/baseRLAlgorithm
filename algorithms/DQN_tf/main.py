from replayMemory import replayMemory
import observation
import gym
from dqn import DQN_agent
import matplotlib.pyplot as plt
from algorithms.DQN_pytorch_offical import plotting
import maze

def random_play_200(env, agent, episode_rewards):
    for i in range(200):
        # at the begining of each episode, reset the env
        env.reset()
        current_state = observation.get_observation(env)
        print(i)

        total_reward = 0
        while True:
            env.render()
            action = agent.select_action(current_state)
            observations, reward, done = env.step(action)

            next_state = observation.get_observation(env)

            experience = (current_state, action, reward, next_state)
            agent.replayMemory.append(current_state, action, reward, next_state)
            # print(len(replayMemory))
            total_reward += reward
            if done:
                break
            else:
                current_state = next_state

        episode_rewards.append(total_reward)
        plotting.plot_durations(episodes_rewards)


if __name__ == "__main__":
    # Initialize replay memory D to capacity N
    replayMemory = replayMemory(10000)
    # Initialize action-value function Q with random weights
    
    env = maze.Maze()
    
    agent = DQN_agent(128, replayMemory, env)

    episodes_rewards = []
    #random_play_200(agent.env, agent, episodes_rewards)
    print("random over!!!!!!")
    # play 100 episodes
    for i in range(300):
        # at the begining of each episode, reset the env
        last_time = 0
        env.reset()
        current_state = observation.get_observation(env)
        print(i)
        total_reward = 0
        while True:
            env.render()
            action = agent.select_action(current_state)
            observations, reward, done = env.step(action)
            total_reward += reward
            next_state = observation.get_observation(env)

            experience = (current_state, action, reward, next_state)
            agent.replayMemory.append(current_state, action, reward, next_state)
            #print(len(replayMemory))
            agent.learn()
            last_time += 1
            if done:
                break
            else:
                current_state = next_state
        episodes_rewards.append(total_reward)
        plotting.plot_durations(episodes_rewards)
        print("The number",i, "episode last", last_time, "times", "and got rewards:", total_reward)


    plt.show()

