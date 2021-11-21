import tensorflow as tf
import matplotlib.pyplot as plt
import numpy as np
import tensorflow_datasets as tfds

from data_preparation import prepare_f_mnist_data


def main():
    # -------- task 1 "Data set" ---------
    ds_train_fmnist, ds_test_fmnist = tfds.load(
        "fashion_mnist",
        split=["train", "test"],
        shuffle_files=True,
        as_supervised=True,
    )

    # -------- task 1.1 "Construct a Data Pipeline" ------------

    train_ds = ds_train_fmnist.apply(prepare_f_mnist_data)
    test_ds = ds_test_fmnist.apply(prepare_f_mnist_data)

    # -------- task 2 "Model" ------------

    convlayer_1: tf.keras.layers.Conv2D = create_conv_layer()
    convlayer_2: tf.keras.layers.Conv2D = create_conv_layer()

    # ---------- task 4 "Visualization" ------------
    # _, ax = plt.subplot(2, 1, sharex=True, figsize=(9, 6))
    #
    # ax[0].plot(train_losses, label="train loss")
    # ax[0].plot(test_losses, label="test loss")
    # ax[0].set(ylabel="Loss", title="Loss")
    # ax[0].legend(loc="upper right")
    #
    # ax[1].plot(train_accuracies, label="train accuracy")
    # ax[1].plot(test_accuracies, label="test accuracy")
    # ax[1].set(ylabel="Accuracy", title="Accuracy")
    # ax[1].legend(loc="upper right")
    #
    # plt.xlabel("Epochs")
    # plt.tight_layout()
    # plt.show()

    input_shape = (16, 16)
    n_outputs = 128

    # create input of shape (1,16,16,1)
    input_img = tf.random.uniform(shape=(1, input_shape[0], input_shape[1], 1))

    # instantiate Conv2D layer with 128 filters with kernel size (16,16),
    # without extra padding
    conv_layer = tf.keras.layers.Conv2D(
        filters=n_outputs,
        kernel_size=input_shape,
        strides=(1, 1),
        padding="valid",
    )
    # instantiate dense layer with 128 outputs
    dense_layer = tf.keras.layers.Dense(n_outputs)

    # flatten input to process it with dense layer
    flatted_input = tf.keras.layers.Flatten(input_img)
    dense_output = dense_layer(flatted_input)

    # process input with conv_layer
    conv_output = conv_layer(input_img)

    # reshape weights from dense layer into shape of conv layer weights such
    # that we can use the same weights for both
    dense_weights = tf.reshape(
        dense_layer.weights[0], shape=conv_layer.weights[0].shape
    )

    dense_bias = tf.reshape(
        dense_layer.weights[1], shape=conv_layer.weights[1].shape
    )

    # assign weights from dense layer to conv layer to show they result in
    # the same output
    conv_layer.weights[0].assign(dense_weights)
    conv_layer.weights[1].assign(dense_bias)

    conv_output = conv_layer(input_img)

    conv_output = tf.reshape(conv_output, shape=dense_output.shape)


def create_conv_layer(
    filter_num=64, ker_size=3, stride=(1, 1), pad="same"
) -> tf.keras.layers.Conv2D:
    return tf.keras.layers.Conv2D(
        filters=filter_num, kernel_size=ker_size, strides=stride, padding=pad
    )


if __name__ == "__main__":
    main()
