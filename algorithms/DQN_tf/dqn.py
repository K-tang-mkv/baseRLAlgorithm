import random

import tensorflow as tf
from replayMemory import Experience
import numpy as np

dqn_conv_model = tf.keras.Sequential(
    [
        tf.keras.layers.Conv2D(32, (5,5), activation='relu', input_shape=(39, 39, 1)),
        tf.keras.layers.Conv2D(64, (5,5), activation='relu'),
        tf.keras.layers.Flatten(),
        tf.keras.layers.Dense(64, activation='relu'),
        tf.keras.layers.Dense(2)
    ]
)

dqn_model = tf.keras.Sequential([
    tf.keras.layers.Dense(8, input_dim=4, activation="relu"),
    tf.keras.layers.Dense(16, activation="relu"),
    # tf.keras.layers.Dense(32, activation="relu"),
    # tf.keras.layers.Dense(16, activation="relu"),
    # tf.keras.layers.Dense(8, activation="relu"),
    tf.keras.layers.Dense(2)
])
class DQN_agent(object):
    def __init__(self, batch_size, replay_memory, env, model=dqn_model, Gamma=0.9):
        super(DQN_agent, self).__init__()
        self.batch_size = batch_size
        self.gamma = Gamma
        self.model = model
        #opt = tf.keras.optimizers.Adam(learning_rate=0.001)
        self.model.compile(optimizer="adam", loss='mse', metrics=["accuracy"])
        self.replayMemory = replay_memory
        self.epsilon = 1
        self.env = env

    def learn(self):
        if len(self.replayMemory) < self.batch_size:
            return

        training_set = self.replayMemory.sample(self.batch_size)
        # below operation will return Transition(state=(s0,s1...), action=(a0,a1...), next_state=(s'0, s'1...), reward=(r0, r1...))
        batch = Experience(*zip(*training_set))
        batch_index = np.arange(self.batch_size)

        current_state_batch = np.array(batch.state)

        self.action_batch = tf.constant(batch.action)
        reward_batch = tf.constant(batch.reward)
        next_state_batch = tf.constant(batch.next_state)

        # calculate Q prediction value
        # predict = self.call(current_state_batch, training=True)
        # q_eval = tf.gather(predict, action_batch, batch_dims=action_batch.ndim)

        # calculate Q-next value and Q_target
        q_target = self.model.predict(next_state_batch)
        q_target[batch_index, self.action_batch] = reward_batch + self.gamma * tf.math.reduce_max(q_target, 1)
        #q_target = q_target.reshape(-1, q_target.shape[1])
        self.model.fit(current_state_batch, q_target)

        self.epsilon = self.epsilon - 0.001 if self.epsilon > \
                                                      0.1 else 0.1

    def select_action(self, state):
        # select an action and execute according to epsilon-greedy policy
        sample = random.random()
        state = state.reshape((1, *state.shape))
        if sample > self.epsilon:
            y = self.model.predict(state)
            return int(tf.argmax(y, 1))
        else:
            return self.env.action_space.sample()
        
if __name__ == "__main__":
    dqn_model.compile(optimizer='SGD',
              loss="mse",
              metrics=['accuracy'])

    x = tf.random.normal((128, 4))
    y = tf.random.uniform((128, 2))
    #ins = tf.random.normal((1,78,78,1))
    dqn_model.fit(x, y)
