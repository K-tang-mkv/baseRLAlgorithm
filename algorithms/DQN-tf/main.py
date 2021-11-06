from replayMemory import replayMemory


if __name__ == "__main__":
    # Initialize replay memory D to capacity N
    replayMemory = replayMemory(10000)
    # Initialize action-value function Q with random weights
    