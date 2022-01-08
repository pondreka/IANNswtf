import tensorflow_datasets as tfds
import tensorflow as tf
import matplotlib.pyplot as plt

# ----------- task 1 "Data set" ------------------
train_ds, test_ds = tfds.load(
    "mnist", split=["train", "test"], as_supervised=True
)


def prepare_mnist_data(mnist, noise: float = 0.1):
    # convert data from uint8 to float32
    mnist = mnist.map(lambda img, target: tf.cast(img, tf.float32))
    # sloppy input normalization, just bringing image values from range
    # [0, 255] to [-1, 1]
    mnist = mnist.map(lambda img: img / tf.norm(img))
    # Expand dimensionality
    mnist = mnist.map(lambda img: tf.expand_dims(img, axis=-1))
    # Add noise to image and create target vector
    mnist = mnist.map(
        lambda img: (
            img
            + tf.random.normal(shape=img.shape, mean=0.0, stddev=1.0 * noise),
            img,
        )
    )
    # Scaling of the noisy image
    mnist = mnist.map(
        lambda img, target: (
            tf.clip_by_value(img, clip_value_min=0, clip_value_max=1),
            target,
        )
    )
    # cache this progress in memory, as there is no need to redo it; it is
    # deterministic after all
    mnist = mnist.cache()
    # shuffle, batch, prefetch
    mnist = mnist.shuffle(1024)
    mnist = mnist.batch(64)
    mnist = mnist.prefetch(128)
    # return preprocessed dataset
    return mnist


train_ds = train_ds.apply(prepare_mnist_data)
test_ds = test_ds.apply(prepare_mnist_data)

for x, t in train_ds:
    n = 10  # How many digits we will display
    plt.figure(figsize=(20, 4))
    for i in range(n):
        # Display noisy image
        ax = plt.subplot(2, n, i + 1)
        plt.imshow(tf.squeeze(x[i]))
        plt.gray()
        ax.get_xaxis().set_visible(False)
        ax.get_yaxis().set_visible(False)

        # Display reconstruction
        ax = plt.subplot(2, n, i + 1 + n)
        plt.imshow(tf.squeeze(t[i]))
        plt.gray()
        ax.get_xaxis().set_visible(False)
        ax.get_yaxis().set_visible(False)
    plt.show()
    break

# ------------ task 2 "Model" --------------
# ------------ task 2.1 "Convolutional Autoencoder" ----------------


class Encoder(tf.keras.Model):
    def __init__(self):
        super(Encoder, self).__init__()

        self.conv_layer_1 = tf.keras.layers.Conv2D(
            filters=16,
            kernel_size=(3, 3),
            activation=tf.keras.activations.relu,
            padding="same",
        )
        self.pooling_layer_1 = tf.keras.layers.MaxPool3D(
            padding="same", strides=(2, 2, 1)
        )
        self.conv_layer_2 = tf.keras.layers.Conv2D(
            filters=8,
            kernel_size=(1, 1),
            activation=tf.keras.activations.relu,
            padding="same",
        )
        self.pooling_layer_2 = tf.keras.layers.MaxPool3D(
            padding="same", strides=(2, 2, 1)
        )

        self.fatten_layer = tf.keras.layers.Flatten()
        self.output_layer = tf.keras.layers.Dense(
            10, activation=tf.keras.activations.relu
        )

    @tf.function
    def call(self, inputs: tf.Tensor) -> tf.Tensor:

        out_conv_1 = self.conv_layer_1(inputs)
        out_pooling_1 = self.pooling_layer_1(out_conv_1)
        out_conv_2 = self.conv_layer_2(out_pooling_1)
        out_pooling_2 = self.pooling_layer_2(out_conv_2)
        out_flatten = self.fatten_layer(out_pooling_2)
        output = self.output_layer(out_flatten)

        return output


class Decoder(tf.keras.Model):
    def __init__(self):
        super(Decoder, self).__init__()

        self.dense_layer = tf.keras.layers.Dense(
            392, activation=tf.keras.activations.relu
        )

        self.reshape_layer = tf.keras.layers.Reshape(target_shape=(7, 7, 1, 8))

        self.conv_transposed_1 = tf.keras.layers.Conv3DTranspose(
            filters=16,
            kernel_size=(1, 1, 1),
            strides=(2, 2, 1),
            padding="same",
            activation=tf.keras.activations.relu,
        )

        self.conv_transposed_2 = tf.keras.layers.Conv3DTranspose(
            filters=16,
            kernel_size=(3, 3, 1),
            strides=(2, 2, 1),
            padding="same",
            activation=tf.keras.activations.relu,
        )

        self.output_layer = tf.keras.layers.Conv2D(
            filters=1,
            kernel_size=(1, 1),
            activation=tf.keras.activations.sigmoid,
            padding="same",
        )

    @tf.function
    def call(self, inputs: tf.Tensor) -> tf.Tensor:
        out_dense = self.dense_layer(inputs)
        out_reshape = self.reshape_layer(out_dense)
        out_conv_trans_1 = self.conv_transposed_1(out_reshape)
        out_conv_trans_2 = self.conv_transposed_2(out_conv_trans_1)
        output = self.output_layer(out_conv_trans_2)

        return output


class Autoencoder(tf.keras.Model):
    def __init__(self):
        super(Autoencoder, self).__init__()

        self.encoder = Encoder()
        self.decoder = Decoder()

    @tf.function
    def call(self, inputs: tf.Tensor) -> tf.Tensor:
        out_encoder = self.encoder(inputs)
        out_decoder = self.decoder(out_encoder)

        return out_decoder


autoencoder = Autoencoder()

# --------------- task 3 " Training" ---------------
import numpy as np


def train_step(
    model: tf.keras.Model,
    data: tf.Tensor,
    target: tf.Tensor,
    loss_function: tf.keras.losses,
    optimizer: tf.keras.optimizers,
):
    """Training iteration over one input data.

    Args:
        model: Model to train.
        data: Data used to calculate the predictions.
        target: Targets for the loss and accuracy calculation.
        loss_function: Function used to calculate the loss.
        optimizer: an optimizer to apply with the gradient

    Returns:
        (tf.Tensor, tf.Tensor): A tuple with the calculated loss and
            accuracy.
    """

    with tf.GradientTape() as tape:
        prediction = model(data)
        loss = loss_function(target, prediction)
        accuracy = tf.keras.backend.argmax(
            target, axis=1
        ) == tf.keras.backend.argmax(prediction, axis=1)
        accuracy = np.mean(accuracy)
        gradients = tape.gradient(loss, model.trainable_variables)
    optimizer.apply_gradients(zip(gradients, model.trainable_variables))

    return loss, accuracy


def test(model: tf.keras.Model, test_data, loss_function: tf.keras.losses):
    """Test iteration over all test data.

    Args:
        :param model: Model to train.
        :param test_data: Dataset to test with the model.
        :param loss_function: Function used to calculate the loss.

    Returns:
        (float, float): A tuple with the calculated loss and accuracy
    """
    test_accuracy_aggregator = []
    test_loss_aggregator = []
    for (data, target) in test_data:
        prediction = model(data)
        sample_test_loss = loss_function(target, prediction)
        sample_test_accuracy = tf.keras.backend.argmax(
            target, axis=1
        ) == tf.keras.backend.argmax(prediction, axis=1)
        sample_test_accuracy = np.mean(sample_test_accuracy)
        test_loss_aggregator.append(sample_test_loss.numpy())
        test_accuracy_aggregator.append(np.mean(sample_test_accuracy))

    test_loss = tf.reduce_mean(test_loss_aggregator)
    test_accuracy = tf.reduce_mean(test_accuracy_aggregator)

    return test_loss, test_accuracy


def training(
    model, loss, num_epochs, optimizer, train_ds, test_ds,
):
    """Train the mode for the number of epochs specified.

    Args:
        model: model to train.
        loss: loss function used for the training and test the model.
        num_epochs: number of iterations for the training.
        optimizer:  Optimizer for the train step.
        train_ds:  training dataset
        test_ds:  testing data set
        init_model:  model to use for the initial testing before the
            training iterations

    Returns:
        ((list, list), (list, list)):
            2 touples, of 2-touples. Each of the three representing
            train and test datasets, for which the two inner
            ones represent their respective accuracy and loss.
    """

    # Prepare some data for the final visualization
    sub_test_ds = list(test_ds)[:10]
    # lists of tensors
    train_losses: list = []
    train_accuracies: list = []
    test_losses: list = []
    test_accuracies: list = []
    # testing train dataset once before we starting the training
    train_loss, train_accuracy = test(model, train_ds, loss)
    train_losses.append(train_loss)
    train_accuracies.append(train_accuracy)
    # testing test dataset once before we starting the training
    test_loss, test_accuracy = test(model, test_ds, loss)
    test_losses.append(test_loss)
    test_accuracies.append(test_accuracy)
    # We train for num_epochs epochs.
    for epoch in range(num_epochs):
        print(
            f"\nEpoch {epoch}: Train accuracy:\t{train_accuracies[-1]}"
            f"\n\t\t Test accuracy:\t\t{test_accuracies[-1]}"
        )

        # training (and checking in with training)
        epoch_loss_agg = []
        epoch_accuracy_agg = []
        for inp, target in train_ds:
            train_loss, train_accuracy = train_step(
                model, inp, target, loss, optimizer
            )
            epoch_loss_agg.append(train_loss)
            epoch_accuracy_agg.append(train_accuracy)

        # track training loss
        train_losses.append(tf.reduce_mean(epoch_loss_agg))
        train_accuracies.append(tf.reduce_mean(epoch_accuracy_agg))

        # testing, so we can track valid accuracy and valid loss
        test_loss, test_accuracy = test(model, test_ds, loss)
        test_losses.append(test_loss)
        test_accuracies.append(test_accuracy)

        # TODO: Implement printing 10 images in each epoch --------------
        # TODO: If the visibility of the numbers improves, take out the
        # accuracy, because it's trash

        decoded_imgs = model(sub_test_ds)

        n = 10  # How many digits we will display
        plt.figure(figsize=(20, 4))
        for i in range(n):
            # Display noisy image
            ax = plt.subplot(2, n, i + 1)
            plt.imshow(tf.squeeze(sub_test_ds[i]))
            plt.gray()
            ax.get_xaxis().set_visible(False)
            ax.get_yaxis().set_visible(False)

            # Display reconstruction
            ax = plt.subplot(2, n, i + 1 + n)
            plt.imshow(tf.squeeze(decoded_imgs[i]))
            plt.gray()
            ax.get_xaxis().set_visible(False)
            ax.get_yaxis().set_visible(False)
        plt.show()

    return (
        (train_accuracies, train_losses),
        (test_accuracies, test_losses),
    )


epochs = 2
learning_rate = 0.001
mse_loss = tf.keras.losses.MeanSquaredError()
adam_optimizer = tf.keras.optimizers.Adam(learning_rate)

train, test = training(
    model=autoencoder,
    loss=mse_loss,
    num_epochs=epochs,
    optimizer=adam_optimizer,
    train_ds=train_ds,
    test_ds=test_ds,
)

autoencoder.summary()
