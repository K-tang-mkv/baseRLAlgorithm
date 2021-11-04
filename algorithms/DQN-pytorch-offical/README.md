# DQN
An abbreviation for Deep Q-Network, which is based on the convolutional neural network. We use a neural network to approximate the **q_value** function, which is the expected return value for a given action **a**. The reason we use a neural network to be our approximation function is that the target function we approximate is non-linear so that the best way to fit the data is to use the universal approximation function **neural network**. 
## Q-network
Q-network is a convolutional neural network that takes in the difference between the current and previous screen patches. It populate some number of outputs as the number of actions **Q(s, a)**, which actually is the expected return value for the given state.

This network implemented is composed of three convolutional layers and one fully connected layer, and we use pytorch to implement this network.
