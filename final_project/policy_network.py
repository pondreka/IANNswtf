import os.path
import numpy as np
import tensorflow as tf
tf.compat.v1.disable_eager_execution()

OBSERVATIONS_SIZE = 6400


class Network:
    def __init__(self, hidden_layer_size, learning_rate, checkpoints_dir):
        self.learning_rate = learning_rate

        self.sess = tf.compat.v1.InteractiveSession()

        self.observations = tf.compat.v1.placeholder(tf.float32,
                                           [None, OBSERVATIONS_SIZE])
        # +1 for up, -1 for down
        self.sampled_actions = tf.compat.v1.placeholder(tf.float32, [None, 1])
        self.advantage = tf.compat.v1.placeholder(
            tf.float32, [None, 1], name='advantage')

        h = tf.compat.v1.layers.dense(
            self.observations,
            units=hidden_layer_size,
            activation=tf.nn.relu,
            kernel_initializer=tf.keras.initializers.glorot_normal())

        self.up_probability = tf.compat.v1.layers.dense(
            h,
            units=1,
            activation=tf.sigmoid,
            kernel_initializer=tf.keras.initializers.glorot_normal())

        # Train based on the log probability of the sampled action.
        #
        # The idea is to encourage actions taken in rounds where the agent won,
        # and discourage actions in rounds where the agent lost.
        # More specifically, we want to increase the log probability of winning
        # actions, and decrease the log probability of losing actions.
        #
        # Which direction to push the log probability in is controlled by
        # 'advantage', which is the reward for each action in each round.
        # Positive reward pushes the log probability of chosen action up;
        # negative reward pushes the log probability of the chosen action down.
        self.loss = tf.compat.v1.losses.log_loss(
            labels=self.sampled_actions,
            predictions=self.up_probability,
            weights=self.advantage)
        optimizer = tf.keras.optimizers.Adam(self.learning_rate)
        self.train_op = optimizer.minimize(self.loss, var_list=[])

        tf.compat.v1.global_variables_initializer().run()

        self.saver = tf.compat.v1.train.Saver()
        self.checkpoint_file = os.path.join(checkpoints_dir,
                                            'policy_network.ckpt')

    def load_checkpoint(self):
        print("Loading checkpoint...")
        self.saver.restore(self.sess, self.checkpoint_file)

    def save_checkpoint(self):
        print("Saving checkpoint...")
        self.saver.save(self.sess, self.checkpoint_file)

    def forward_pass(self, observations):
        up_probability = self.sess.run(
            self.up_probability,
            feed_dict={self.observations: observations.reshape([1, -1])})
        return up_probability

    def train(self, state_action_reward_tuples):
        print("Training with %d (state, action, reward) tuples" %
              len(state_action_reward_tuples))

        states, actions, rewards = zip(*state_action_reward_tuples)
        states = np.vstack(states)
        actions = np.vstack(actions)
        rewards = np.vstack(rewards)

        feed_dict = {
            self.observations: states,
            self.sampled_actions: actions,
            self.advantage: rewards
        }
        self.sess.run(self.train_op, feed_dict)