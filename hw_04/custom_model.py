import tensorflow as tf

# ---------- task 2 "Model" -----------


# Definition of a custom layer for our model.
class CustomLayer(tf.keras.layers.Layer):
    """A custom dense layer."""

    def __init__(self, units=64, activation=tf.nn.sigmoid):
        super(CustomLayer, self).__init__()
        self.units = units
        self.activation = activation

    def build(self, input_shape):
        self.w = self.add_weight(
            shape=(input_shape[-1], self.units),
            initializer="random_normal",
            trainable=True,
        )
        self.b = self.add_weight(
            shape=(self.units,), initializer="random_normal", trainable=True,
        )

    @tf.function
    def call(self, inputs):
        incoming_inputs = tf.matmul(inputs, self.w) + self.b
        return self.activation(incoming_inputs)


# Definition of a custom model by using the custom layers.
class CustomModel(tf.keras.Model):
    """Simple custom Model definition (MLP)."""

    def __init__(self):
        super(CustomModel, self).__init__()
        self.hidden_layer1 = CustomLayer()
        self.hidden_layer2 = CustomLayer()
        self.output_layer = CustomLayer(units=1)

    @tf.function
    def call(self, inputs):
        output_of_hl_1 = self.hidden_layer1(inputs)
        output_of_hl_2 = self.hidden_layer2(output_of_hl_1)
        return self.output_layer(output_of_hl_2)
