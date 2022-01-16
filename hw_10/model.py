import tensorflow as tf


# ------------- task 2.2 "Model" -------------------
class SkipGram(tf.keras.layers.Layer):
    """A custom SkipGram layer"""

    def __init__(self, units: int, activation=tf.nn.sigmoid):
        """Constructor

        :param units: Input units.
        :param activation: Activation function used in the forward pass.
        """
        super(SkipGram, self).__init__()

         # TODO: Initialize vocabulary and embedding size.

    def build(self, input_shape: tf.TensorShape) -> None:
        """Instantiation of weights and bias

        :param input_shape: shape for weights and bias creation.
        """

        # TODO: initialize the embedding and score matrices of correct shape by using vocabulary and embedding size.
        self.score_matrices = self.add_weight(
            shape=(input_shape[-1], self.units),
            initializer="random_normal",
            trainable=True,
        )

    @tf.function
    def call(self, inputs: tf.Tensor) -> tf.Tensor:
        """Forward propagation

        :param inputs: Inputs from layer.
        """
        # TODO: get the embeddings using tf.nn.embedding lookup()

        # TODO: calculate and return the loss using tf.nn.nce_loss
        return self.activation(incoming_inputs)


class Generator(tf.keras.Model):

    def __init__(self):
        super(Generator, self).__init__()

        learning_rate = 0.01
        self.optimizer = tf.keras.optimizers.Adam(learning_rate)

        self.fake_loss = tf.keras.losses.BinaryCrossentropy()

        self.loss_metric = tf.keras.metrics.Mean(name="g_loss")

        l2_lambda = 0.01
        dropout_amount = 0.5

        self.all_layers = [
            tf.keras.layers.Reshape((10, 10, 1), input_shape=(100,)),

            tf.keras.layers.Conv2DTranspose(
                filters=32,
                kernel_size=(6, 6),
                strides=(1, 1),
                padding="valid",
                activation=None,
                kernel_initializer=tf.keras.initializers.glorot_uniform,
                kernel_regularizer=tf.keras.regularizers.L2(l2_lambda)
            ),
            tf.keras.layers.BatchNormalization(),
            tf.keras.layers.Activation(tf.nn.relu),

            tf.keras.layers.Dropout(dropout_amount),

            tf.keras.layers.Conv2DTranspose(
                filters=32,
                kernel_size=(4, 4),
                strides=(1, 1),
                padding="valid",
                activation=None,
                kernel_initializer=tf.keras.initializers.glorot_uniform,
                kernel_regularizer=tf.keras.regularizers.L2(l2_lambda)
            ),
            tf.keras.layers.BatchNormalization(),
            tf.keras.layers.Activation(tf.nn.relu),

            tf.keras.layers.Dropout(dropout_amount),

            tf.keras.layers.Conv2DTranspose(
                filters=32,
                kernel_size=(2, 2),
                strides=(1, 1),
                padding="valid",
                activation=None,
                kernel_initializer=tf.keras.initializers.glorot_uniform,
                kernel_regularizer=tf.keras.regularizers.L2(l2_lambda)
            ),
            tf.keras.layers.BatchNormalization(),
            tf.keras.layers.Activation(tf.nn.relu),

            tf.keras.layers.Dropout(dropout_amount),

            tf.keras.layers.Conv2D(
                filters=1,
                kernel_size=(1, 1),
                strides=(1, 1),
                padding="valid",
                activation=None,
                kernel_initializer=tf.keras.initializers.glorot_uniform,
                kernel_regularizer=tf.keras.regularizers.L2(l2_lambda)
            ),
            tf.keras.layers.BatchNormalization(),
            tf.keras.layers.Activation(tf.nn.tanh)
        ]

    @tf.function
    def call(self, x, training: bool = False):

        for layer in self.all_layers:
            try:
                x = layer(x, training)
            except:
                x = layer(x)

        return x

    def loss_function(self, fake_data_pred):
        loss = self.fake_loss(tf.ones_like(fake_data_pred), fake_data_pred)

        return loss

    def reset_metrics(self):
        self.loss_metric.reset_states()
