
import tensorflow as tf


# ---------- task 2 "Model" -----------

class CustomLayer(tf.keras.layers.Layer):
    """A custom dense layer """

    def __init__(self, units: int, activation=tf.nn.sigmoid):
        """ Constructor

        :param units: Input units.
        :param activation: Activation function used in the forward pass.
        """
        super(CustomLayer, self).__init__()
        self.units = units
        self.activation = activation

    def build(self, input_shape: tf.TensorShape) -> None:
        """ Instantiation of weights and bias

        :param input_shape: shape for weights and bias creation.
        """
        self.w = self.add_weight(
            shape=(input_shape[-1], self.units),
            initializer="random_normal",
            trainable=True,
            regularizer=tf.keras.regularizers.L1L2()
        )
        self.b = self.add_weight(
            shape=(self.units,),
            initializer="random_normal",
            trainable=True,
        )

    @tf.function
    def call(self, inputs: tf.Tensor) -> tf.Tensor:
        """ Forward propagation

        :param inputs: Inputs from layer.
        """
        incoming_inputs = tf.matmul(inputs, self.w) + self.b
        return self.activation(incoming_inputs)


class DropoutModel(tf.keras.Model):
    """ Definition of a custom Dropput Model """

    def __init__(self, rate=0.1):
        """ Constructor """
        super(DropoutModel, self).__init__()
        self.hidden_layer1 = CustomLayer(units=64)
        self.dropout_layer = tf.keras.layers.Dropout(rate=rate)
        self.hidden_layer2 = CustomLayer(units=64)
        self.output_layer = CustomLayer(units=1)
        # Dropout Layer

    @tf.function
    def call(self, inputs: tf.Tensor) -> tf.Tensor:
        """ Model's forward pass through instantiated layers

        :param inputs: inputs to feed the model
        """
        output_of_hl_1 = self.hidden_layer1(inputs)
        # Dropout between the hidden layers
        output_of_dl_1 = self.dropout_layer(output_of_hl_1)
        output_of_hl_2 = self.hidden_layer2(output_of_dl_1)
        return self.output_layer(output_of_hl_2)
