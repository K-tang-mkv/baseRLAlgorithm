import maze
from agent import QLAgent

if __name__ == "__main__":
    env = maze.Maze()
    agent = QLAgent(env)

    for i in range(100):
        current_observation = str(env.reset())
        count = 0
        total_reward = 0
        while True:
            env.render()
            action = agent.select_action(current_observation)
            next_observation, reward, done = env.step(action)
            agent.learn(current_observation, action, str(next_observation), reward)
            total_reward += reward
            count += 1
            if done:
                break
        print("episode", i, "take", count, "times", "and got", total_reward)