import numpy as np
import tensorflow as tf
import tensorflow_datasets as tfds


# --------- task 1 "Data Set" ---------------


def one_hot_converter(tensor):
    vocab = {"A": "1", "C": "2", "G": "3", "T": "0"}
    for key in vocab.keys():
        tensor = tf.strings.regex_replace(tensor, key, vocab[key])
    split = tf.strings.bytes_split(tensor)
    labels = tf.cast(tf.strings.to_number(split), tf.uint8)
    onehot = tf.one_hot(labels, 4)
    onehot = tf.reshape(onehot, (-1,))
    return onehot


def prepare_ds(dataset, take_amount: int = 100000):
    dataset = dataset.take(take_amount)
    dataset = dataset.map(
        lambda seq, label: (one_hot_converter(seq), tf.one_hot(label, 10))
    )
    dataset = dataset.cache()
    dataset = dataset.shuffle(1000)
    dataset = dataset.batch(32)
    dataset = dataset.prefetch(10)
    return dataset


def prepare_ds_thousand(dataset):
    return prepare_ds(dataset, 1000)


train_ds, test_ds = tfds.load(
    "genomics_ood",
    split=["train", "test"],
    shuffle_files=True,
    as_supervised=True,
)
train_ds = train_ds.apply(prepare_ds)
test_ds = test_ds.apply(prepare_ds_thousand)

for elem in train_ds.take(1):
    print(elem)

# ---------- task 2 "Model" -----------
# first we define a custom layer


class CustomLayer(tf.keras.layers.Layer):
    """A custom dense layer."""

    def __init__(self, units=256, activation=tf.nn.sigmoid):
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

    def call(self, inputs):
        incoming_inputs = tf.matmul(inputs, self.w) + self.b
        return self.activation(incoming_inputs)


# then the model itself
class CustomModel(tf.keras.Model):
    """Simple custom Model definition (MLP)."""

    def __init__(self):
        super(CustomModel, self).__init__()
        self.hidden_layer1 = CustomLayer()
        self.hidden_layer2 = CustomLayer()
        self.output_layer = CustomLayer(units=10, activation=tf.nn.softmax)

    def call(self, inputs):
        output_of_hl_1 = self.hidden_layer1(inputs)
        output_of_hl_2 = self.hidden_layer2(output_of_hl_1)
        return self.output_layer(output_of_hl_2)


# --------- task 3 "Training" ---------------


def train_step(model, input, target, loss_function, optimizer):
    # loss_object and optimizer_bject are instances of respective
    # tensorflow classes
    with tf.GradientTape() as tape:
        prediction = model(input)
        loss = loss_function(target, prediction)
        gradients = tape.gradient(loss, model.trainable_variables)
    optimizer.apply_gradients(zip(gradients, model.trainable_variables))
    return loss


def test(model, test_data, loss_function):
    # test over complete test data
    test_accuracy_aggregator = []
    test_loss_aggregator = []
    for (input, target) in test_data:
        prediction = model(input)
        sample_test_loss = loss_function(target, prediction)
        sample_test_accuracy = np.argmax(target, axis=1) == np.argmax(
            prediction, axis=1
        )
        sample_test_accuracy = np.mean(sample_test_accuracy)
        test_loss_aggregator.append(sample_test_loss.numpy())
        test_accuracy_aggregator.append(np.mean(sample_test_accuracy))

    test_loss = tf.reduce_mean(test_loss_aggregator)
    test_accuracy = tf.reduce_mean(test_accuracy_aggregator)

    return test_loss, test_accuracy


learn_rate = 0.1
cross_entropy_loss = tf.keras.losses.CategoricalCrossentropy()
num_epochs = 10

super_model = CustomModel()
optimizer = tf.keras.optimizers.SGD(learn_rate)

# Prepare some data for the final visualization
# lists of tensors
train_losses: list = []

test_losses: list = []
test_accuracies: list = []


# testing once before we begin
test_loss, test_accuracy = test(super_model, test_ds, cross_entropy_loss)
test_losses.append(test_loss)
test_accuracies.append(test_accuracy)

# check how model performs on train data once before we begin
train_loss, _ = test(super_model, train_ds, cross_entropy_loss)
train_losses.append(train_loss)

# We train for num_epochs epochs.
for epoch in range(num_epochs):
    print(f"Epoch: {str(epoch)} starting with accuracy {test_accuracies[-1]}")

    # training (and checking in with training)
    epoch_loss_agg = []
    for input, target in train_ds:
        train_loss = train_step(
            super_model, input, target, cross_entropy_loss, optimizer
        )
        epoch_loss_agg.append(train_loss)

    # track training loss
    train_losses.append(tf.reduce_mean(epoch_loss_agg))

    # testing, so we can track accuracy and test loss
    test_loss, test_accuracy = test(super_model, test_ds, cross_entropy_loss)
    test_losses.append(test_loss)
    test_accuracies.append(test_accuracy)
