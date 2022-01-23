import tensorflow as tf


# ------------- task 2.2 "Model" -------------------
class SkipGram(tf.keras.layers.Layer):
    """A custom SkipGram layer"""

    def __init__(self, vocabulary_size, embedding_size):
        """ Constructor """
        super(SkipGram, self).__init__()

        # Initialize vocabulary and embedding size.
        self.vocabulary_size = vocabulary_size
        self.embedding_size = embedding_size

    def build(self, input_shape) -> None:
        """Instantiation of weights and bias

        :param input_shape: shape for weights and bias creation.
        """

        # initialize the embedding and score matrices of correct shape by using vocabulary and embedding size.
        self.embedding_weights = self.add_weight(
            shape=(input_shape[-1], self.embedding_size),
            initializer='random_normal')

        self.score_weights = self.add_weight(
            shape=(self.embedding_size, self.vocabulary_size),
            initializer='random_normal')

        self.score_bias = self.add_weight(
            shape=(self.vocabulary_size,),
            initializer='random_normal')

    @tf.function
    def call(self, inputs: tf.Tensor, target: tf.Tensor) -> tf.Tensor:
        """Forward propagation

        :param inputs: Inputs for layer.
        """

        # get the embeddings using tf.nn.embedding_lookup()
        # TODO: are those the embedding_weights or the weights between the embedding and the score?
        # TODO: maybe we need to one-hot encode the words (embedding_lookup requires int and not string)
        embedding = tf.nn.embedding_lookup(self.embedding_weights, inputs)

        # TODO: Manually average over the batches
        # calculate and return the loss using tf.nn.nce_loss
        loss = tf.nn.nce_loss(
            weights=self.score_weights,
            biases=self.score_bias,
            labels=target,
            inputs=embedding,
            num_classes=self.vocabulary_size,
        )

        return embedding, loss
