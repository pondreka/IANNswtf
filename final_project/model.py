import numpy as np
import tensorflow as tf


class MainModel(tf.keras.Model):
    """Model that will be contained in the DQN."""

    def __init__(self, num_states, hidden_units, num_actions):
        """Constructor for class."""
        super().__init__()
        self.input_layer = tf.keras.layers.InputLayer(
            input_shape=(num_states,)
        )
        self.hidden_layers = []
        for i in hidden_units:
            self.hidden_layers.append(
                tf.keras.layers.Dense(
                    i, activation="tanh", kernel_initializer="RandomNormal"
                )
            )
        self.output_layer = tf.keras.layers.Dense(
            num_actions, activation="linear", kernel_initializer="RandomNormal"
        )

    @tf.function
    def call(self, inputs):
        """Main functionality of model is through this function.

        Pass the inputs through all the hidden layers up until the output
        layers, where it will also be returned.
        """
        output = self.input_layer(inputs)
        for layer in self.hidden_layers:
            output = layer(output)
        output = self.output_layer(output)
        return output


class DQN:
    """Class definition for the Deep Q-Network for RL."""

    def __init__(
        self,
        num_states,
        num_actions,
        hidden_units,
        gamma,
        max_experiences,
        min_experiences,
        batch_size,
        lr,  # the lean rate
    ):
        self.num_actions = num_actions
        self.batch_size = batch_size
        self.optimizer = tf.optimizers.Adam(lr)
        # discount rate: [0, 1) so not too much reward is given
        self.gamma = gamma
        self.model = MainModel(num_states, hidden_units, num_actions)
        # types of experience to save:
        # States, Actions, Rewards, NextState, Done/Goal
        self.experience = {"s": [], "a": [], "r": [], "s2": [], "done": []}
        # max amount of experience to keep track off. After this amount,
        # the oldest ones shall be deleted.
        self.max_experiences = max_experiences
        # Minimum amount of experience required to learn.
        self.min_experiences = min_experiences

    def predict(self, inputs):
        """Attempt to predict outcome based on intern model.

        Input is a batch, with dimensions > 1 (even if single input).
        """
        return self.model(np.atleast_2d(inputs.astype("float32")))

    def train(self, TargetNet):
        """Model training function to call each epoch."""
        if len(self.experience["s"]) < self.min_experiences:
            return 0
        ids = np.random.randint(
            low=0, high=len(self.experience["s"]), size=self.batch_size
        )
        states = np.asarray([self.experience["s"][i] for i in ids])
        actions = np.asarray([self.experience["a"][i] for i in ids])
        rewards = np.asarray([self.experience["r"][i] for i in ids])
        states_next = np.asarray([self.experience["s2"][i] for i in ids])
        dones = np.asarray([self.experience["done"][i] for i in ids])
        value_next = np.max(TargetNet.predict(states_next), axis=1)
        actual_values = np.where(
            dones, rewards, rewards + self.gamma * value_next
        )
        with tf.GradientTape() as tape:
            selected_action_values = tf.math.reduce_sum(
                self.predict(states) * tf.one_hot(actions, self.num_actions),
                axis=1,
            )
            loss = tf.math.reduce_mean(
                tf.square(actual_values - selected_action_values)
            )
        variables = self.model.trainable_variables
        gradients = tape.gradient(loss, variables)
        self.optimizer.apply_gradients(zip(gradients, variables))
        return loss

    def get_action(self, states, epsilon):
        """Determine what action gives most reward, or random.

        From a random set of states, depending on how much epsilon
        has decayed [0, 1], choose either the best known action (exploit),
        or choose a random one out of the possible ones (explore).
        """
        if np.random.random() < epsilon:
            return np.random.choice(self.num_actions)
        else:
            return np.argmax(self.predict(np.atleast_2d(states))[0])

    def add_experience(self, exp):
        """Enter exp into the experience FIFO queue."""
        if len(self.experience["s"]) >= self.max_experiences:
            for key in self.experience:
                self.experience[key].pop(0)
        for key, value in exp.items():
            self.experience[key].append(value)

    def copy_weights(self, TrainNet):
        """Helper funciton to help clone the network."""
        variables1 = self.model.trainable_variables
        variables2 = TrainNet.model.trainable_variables
        for v1, v2 in zip(variables1, variables2):
            v1.assign(v2.numpy())
