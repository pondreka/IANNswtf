import tensorflow as tf


# ---------- task 2 "Model" -----------

# Definition of a custom layer for our model.
class CustomLayer(tf.keras.layers.Layer):
    """A custom dense layer"""

    def __init__(self, units: int, activation=tf.nn.sigmoid):
        """Constructor

        Args:
          units: Input units.
          activation: Activation function used in the forward pass.
        """
        super(CustomLayer, self).__init__()
        self.units = units
        self.activation = activation

    def build(self, input_shape: tf.TensorShape) -> None:
        """Instantiation of weights and bias

        Args:
          input_shape: shape for weights and bias creation.
        """
        self.w = self.add_weight(
            shape=(input_shape[-1], self.units),
            initializer="random_normal",
            trainable=True,
        )
        self.b = self.add_weight(
            shape=(self.units,),
            initializer="random_normal",
            trainable=True,
        )

    @tf.function
    def call(self, inputs: tf.Tensor) -> tf.Tensor:
        """Forward propagation

        Args:
           inputs: Inputs from layer.
        """
        incoming_inputs = tf.matmul(inputs, self.w) + self.b
        return self.activation(incoming_inputs)


class CustomModel(tf.keras.Model):
    """Custom Model with convolutional and pooling layers."""

    def __init__(self):
        """Constructor"""
        super(CustomModel, self).__init__()
        self.hidden_layer1 = CustomLayer(units=64)
        self.hidden_layer2 = CustomLayer(units=64)
        self.output_layer = CustomLayer(units=1)

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

        self.output_layer = CustomLayer(units=10, activation=tf.nn.softmax)

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
        flattened_output_of_pool_2 = tf.keras.layers.Flatten()(
            output_of_pool_2
        )
        return self.output_layer(flattened_output_of_pool_2)
