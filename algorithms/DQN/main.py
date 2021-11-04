from expectedPolicy import select_action
from plotting import plot_durations
import gym
import matplotlib.pyplot as plt

if __name__ == '__main__':
    env = gym.make('CartPole-v1')

    episode_duration = []
    for i in range(50):
        state = env.reset()
        count = 0
        while True:
            count += 1
            action = select_action(state)
            state, _, done, _ = env.step(action)

            if done:
                break
        episode_duration.append(count)
        plot_durations(episode_duration)
    plt.show()