import tensorflow as tf


# ---------- task 2 "Model" -----------


class CustomModel(tf.keras.Model):
    """Custom Model with convolutional and pooling layers."""

    def __init__(self):
        """Constructor"""
        super(CustomModel, self).__init__()
        self.hidden_layer1 = tf.keras.layers.Dense(64)
        self.hidden_layer2 = tf.keras.layers.Dense(64)
        self.output_layer = tf.keras.layers.Dense(1)

    @tf.function
    def call(self, inputs: tf.Tensor) -> tf.Tensor:
        """Model's forward pass through instantiated layers

        Args:
           inputs: inputs to feed the model
        """

        output_of_hl_1 = self.hidden_layer1(inputs)
        output_of_hl_2 = self.hidden_layer2(output_of_hl_1)
        return self.output_layer(output_of_hl_2)


class ConvModel(tf.keras.Model):
    """Custom Model with convolutional and pooling layers."""

    def __init__(self):
        """Constructor"""
        super(ConvModel, self).__init__()

        self.conv_layer_1 = tf.keras.layers.Conv2D(
            filters=16,
            kernel_size=(3, 3),
            activation=tf.keras.activations.relu,
            input_shape=(28, 28, 1),
            padding="same",
        )
        self.pooling_layer_1 = tf.keras.layers.MaxPooling2D(padding="same",)
        self.conv_layer_2 = tf.keras.layers.Conv2D(
            filters=32,
            kernel_size=(3, 3),
            activation=tf.keras.activations.relu,
            padding="same",
        )
        self.flatten_layer = tf.keras.layers.Flatten()
        self.dropout_layer = tf.keras.layers.Dropout(rate=0.7)
        self.output_layer = tf.keras.layers.Dense(
            10, activation=tf.keras.activations.softmax
        )

    @tf.function
    def call(self, inputs: tf.Tensor) -> tf.Tensor:
        """Model's forward pass through instantiated layers

        Args:
           inputs: inputs to feed the model
        """

        output_of_conv_1 = self.conv_layer_1(inputs)
        output_of_pool_1 = self.pooling_layer_1(output_of_conv_1)
        output_of_conv_2 = self.conv_layer_2(output_of_pool_1)
        output_of_flattened = self.flatten_layer(output_of_conv_2)
        output_of_dropout = self.dropout_layer(output_of_flattened)
        return self.output_layer(output_of_dropout)
