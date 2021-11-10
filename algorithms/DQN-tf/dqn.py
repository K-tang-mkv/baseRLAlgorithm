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
        self.optimizer.apply_gradients(zip(gradients, trainable_vars))
        # Update metrics (includes the metric that tracks the loss)
        self.compiled_metrics.update_state(y, y_pred)
        # Return a dict mapping metric names to current value
        return {m.name: m.result() for m in self.metrics}
        
    #def select_action(self, experience):
        

