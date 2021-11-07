import tensorflow as tf
import tensorflow_datasets as tfds


# --------- task 1 "Data Set" ---------------


@tf.function
def one_hot_converter(tensor):
    vocab = {"A": "1", "C": "2", "G": "3", "T": "0"}
    for key in vocab.keys():
        tensor = tf.strings.regex_replace(tensor, key, vocab[key])
    split = tf.strings.bytes_split(tensor)
    labels = tf.cast(tf.strings.to_number(split), tf.uint8)
    onehot = tf.one_hot(labels, 4)
    onehot = tf.reshape(onehot, (-1,))
    return onehot


train_ds = tfds.load(
    "genomics_ood", split="train", shuffle_files=True, as_supervised=True
)
test_ds = tfds.load(
    "genomics_ood", split="test", shuffle_files=True, as_supervised=True
)
train_ds = train_ds.take(100000)
test_ds = test_ds.take(1000)
train_ds = train_ds.map(
    lambda seq, label: (one_hot_converter(seq), tf.one_hot(label, 10))
)
test_ds = test_ds.map(
    lambda seq, label: (one_hot_converter(seq), tf.one_hot(label, 10))
)
train_ds = train_ds.batch(32).prefetch(10)
test_ds = test_ds.batch(32).prefetch(10)


# --------- task 2 "Model" ---------------
# first we define a custom layer


class CustomLayer(tf.keras.layers.Layer):
    """A custom dense layer."""

    def __init__(self, units=256, activation=tf.nn.sigmoid):
        super().__init__()
        self.units = units
        self.activation = activation

    @tf.function
    def build(self, input_shape):
        self.weights = self.add_weight(
            shape=(input_shape[-1], self.units),
            initializer="random_normal",
            trainable=True,
        )
        self.bias = self.add_weight(
            shape=(self.units), initializer="random_normal", traiable=True,
        )

    @tf.function
    def call(self, inputs):
        incoming_inputs = tf.matmul(inputs, self.weights) + self.bias
        return self.activation(incoming_inputs)


# then the model itself
class CustomModel(tf.keras.Model):
    """Simple custom Model definition (MLP)."""

    def __init__(self):
        super().__init__()
        self.hidden_layer1 = CustomLayer()
        self.hidden_layer2 = CustomLayer()
        self.output_layer = CustomLayer(units=10, activation=tf.nn.softmax)

    @tf.function
    def call(self, inputs):
        output_of_hl_1 = self.hidden_layer1(inputs)
        output_of_hl_2 = self.hidden_layer2(output_of_hl_1)
        return self.output_layer(output_of_hl_2)


# --------- task 3 "Training" ---------------
