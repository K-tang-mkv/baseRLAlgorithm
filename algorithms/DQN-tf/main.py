from replayMemory import replayMemory
from object_image import get_screen
import gym
from dqn import DQN, dqn_model
if __name__ == "__main__":
    # Initialize replay memory D to capacity N
    replayMemory = replayMemory(10000)
    # Initialize action-value function Q with random weights
    
    env = gym.make('CartPole-v1')
    
    agent = DQN(dqn_model, 128)
    agent.compile(optimizer='rmsprop', loss='mse')

    # play 100 episodes
    for i in range(100):
        # at the begining of each episode, reset the env
        env.reset()
        current_state = get_screen(env)
        print(i)
        total_reward = 0
        while True:
            action = 0
            _, reward, done, _ = env.step(action)
            total_reward += reward
            next_state = get_screen(env)

            experience = (current_state, action, reward, next_state)
            replayMemory.append(current_state, action, reward, next_state)
            print(len(replayMemory))
            agent.train_step(replayMemory)
            if done:
                break
            else:
                current_state = next_state
