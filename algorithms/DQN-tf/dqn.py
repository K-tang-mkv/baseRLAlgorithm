import random

import tensorflow as tf
from replayMemory import Experience

dqn_model = tf.keras.Sequential(
    [
        tf.keras.layers.Conv2D(32, (3,3), activation='relu', input_shape=(78, 78, 1)),
        tf.keras.layers.Conv2D(64, (3,3), activation='relu'),
        tf.keras.layers.Flatten(),
        tf.keras.layers.Dense(64, activation='relu'),
        tf.keras.layers.Dense(2)
    ]
)

class Dqn(tf.keras.Model):
    def __init__(self, num_actions, env):
        super(Dqn, self).__init__()
        self.env = env
        self.gamma = 0.9
        self.epsilon = 1 # 1~0.1
        self.step = 0
        self.first_Conv2d = tf.keras.layers.Conv2D(32, (3,3), activation='relu', input_shape=(78, 78, 1))
        self.second_Conv2d = tf.keras.layers.Conv2D(64, (3,3), activation='relu')
        self.flatten = tf.keras.layers.Flatten()
        self.dense1 = tf.keras.layers.Dense(64, activation='relu')
        self.dense_output = tf.keras.layers.Dense(num_actions)


    def call(self, inputs, training=None, mask=None):
        conv1_x = self.first_Conv2d(inputs)
        conv2_x = self.second_Conv2d(conv1_x)
        flatten_x = self.flatten(conv2_x)
        dense1_x = self.dense1(flatten_x)
        ouput_x = self.dense_output(dense1_x)

        return ouput_x

    def get_value(self, replayMemory):
        if len(replayMemory) < 128:
            return

        training_set = replayMemory.sample(128)
        # below operation will return Transition(state=(s0,s1...), action=(a0,a1...), next_state=(s'0, s'1...), reward=(r0, r1...))
        batch = Experience(*zip(*training_set))

        current_state_batch = tf.constant(batch.state)
        self.action_batch = tf.constant(batch.action)
        reward_batch = tf.constant(batch.reward)
        next_state_batch = tf.constant(batch.next_state)

        # calculate Q prediction value
        # predict = self.call(current_state_batch, training=True)
        # q_eval = tf.gather(predict, action_batch, batch_dims=action_batch.ndim)

        # calculate target Q-a value
        target = self.call(next_state_batch)
        q_target = reward_batch + self.gamma * tf.math.reduce_max(target, 1)

        return current_state_batch, tf.reshape(q_target, (q_target.shape[0], 1))

    def train_step(self, data):

        current_state_batch, y= data
        with tf.GradientTape() as tape:
            predict = self(current_state_batch, training=True)

            y_pred = tf.gather(predict, self.action_batch, batch_dims=self.action_batch.ndim)
            # Compute the loss value
            # (the loss function is configured in `compile()`)
            loss = self.compiled_loss(y, y_pred, regularization_losses=self.losses)

        # Compute gradients
        trainable_vars = self.trainable_variables
        gradients = tape.gradient(loss, trainable_vars)
        # Update weights
        self.optimizer.apply_gradients(zip(gradients, trainable_vars))
        # Update metrics (includes the metric that tracks the loss)
        self.compiled_metrics.update_state(y, y_pred)
        # Return a dict mapping metric names to current value
        return {m.name: m.result() for m in self.metrics}

    def select_action(self, state):
        # select an action and execute according to epsilon-greedy policy
        sample = random.random()
        self.step += 1
        eps_threshold = self.epsilon - 0.01 * self.step
        state = tf.reshape(state, (1, *state.shape))
        if sample > eps_threshold:
            y = self.predict(state)
            return int(tf.argmax(y, 1))
        else:
            return self.env.action_space.sample()


class DQN(tf.keras.Model):
    def __init__(self, dqnmodel, batch_size, Gamma=0.9):
        super(DQN, self).__init__()
        self.batch_size = batch_size
        self.gamma = Gamma
        self.model = dqnmodel


    def train_step(self, replayMemory):
        if len(replayMemory) < self.batch_size:
            return        
        
        training_set = replayMemory.sample(self.batch_size)
        # below operation will return Transition(state=(s0,s1...), action=(a0,a1...), next_state=(s'0, s'1...), reward=(r0, r1...))
        batch = Experience(*zip(*training_set))

        current_state_batch = tf.constant(batch.state)
        action_batch = tf.constant(batch.action)
        reward_batch = tf.constant(batch.reward)
        next_state_batch = tf.constant(batch.next_state)
        
        # calculate Q prediction value
        predict = self.model(current_state_batch)
        q_eval = tf.gather(predict, action_batch, batch_dims=action_batch.ndim)
        

        # calculate target Q-a value
        target = self.model(next_state_batch)
        q_target = reward_batch + self.gamma * tf.math.reduce_max(target, 1)

        with tf.GradientTape() as tape:
            y_pred = q_eval 
            y = q_target
            # Compute the loss value
            # (the loss function is configured in `compile()`)
            loss = self.compiled_loss(y, y_pred, regularization_losses=self.losses)

        # Compute gradients
        trainable_vars = self.model.trainable_variables
        gradients = tape.gradient(loss, trainable_vars)
        # Update weights
        self.model.optimizer.apply_gradients(zip(gradients, trainable_vars))
        # Update metrics (includes the metric that tracks the loss)
        self.model.compiled_metrics.update_state(y, y_pred)
        # Return a dict mapping metric names to current value
        return {m.name: m.result() for m in self.metrics}
        
    #def select_action(self, experience):
        
if __name__ == "__main__":
    dqn_model.compile(optimizer='adam',
              loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
              metrics=['accuracy'])
    x = tf.random.normal((128, 78, 78, 1))
    y = tf.random.uniform((128, 1))
    ins = tf.random.normal((78,78,1))
    dqn_model.predict(ins)
