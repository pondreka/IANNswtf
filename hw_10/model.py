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

