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

    def __init__(self, input_shape):
        """Constructor"""
        super(ConvModel, self).__init__()

        # TODO: Adjust the proper parameters for each layer
        self.conv_layer_1 = tf.keras.layers.Conv2D(
            filters=64,
            kernel_size=input_shape,
            strides=(1, 1),
            padding="same",  # maybe use "same" instead, since "valid" makes it weird
        )
        self.pooling_layer_1 = tf.keras.layers.MaxPooling2D(
            strides=2, padding="same"
        )
        self.conv_layer_2 = tf.keras.layers.Conv2D(
            filters=32,
            kernel_size=input_shape,
            strides=(1, 1),
            padding="same",
        )
        self.pooling_layer_2 = tf.keras.layers.MaxPooling2D(
            strides=2, padding="same"
        )

        self.output_layer = tf.keras.layers.Dense(10, activation=tf.nn.softmax)

    @tf.function
    def call(self, inputs: tf.Tensor) -> tf.Tensor:
        """Model's forward pass through instantiated layers

        Args:
           inputs: inputs to feed the model
        """

        output_of_conv_1 = self.conv_layer_1(inputs)
        output_of_pool_1 = self.pooling_layer_1(output_of_conv_1)
        output_of_conv_2 = self.conv_layer_2(output_of_pool_1)
        output_of_pool_2 = self.pooling_layer_2(output_of_conv_2)
        # TODO: find a better way of reshaping
        flattened_output_of_pool_2 = tf.keras.layers.Flatten()(
            output_of_pool_2
        )
        # This doesn't work:
        # flattened_output_of_pool_2 = tf.reshape(output_of_pool_2, [-1])
        return self.output_layer(flattened_output_of_pool_2)
