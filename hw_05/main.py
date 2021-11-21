import tensorflow as tf
import matplotlib.pyplot as plt
import numpy as np
import tensorflow_datasets as tfds

# -------- task 1 "Data set" ---------
ds_train_fmnist, ds_test_fmnist = tfds.load(
    "fashion_mnist",
    split=["train", "test"],
    shuffle_files=True,
    as_supervised=True,
)


# -------- task 1.1 "Construct a Data Pipeline" ------------

# TODO: Flatten input?
def prepare_fmnist_data(fmnist):
    # convert data from uint8 to float32
    fmnist = fmnist.map(lambda img, target: (tf.cast(img, tf.float32), target))
    # sloppy input normalization, just bringing image values from range
    # [0, 255] to [-1, 1]
    fmnist = fmnist.map(lambda img, target: ((img / tf.norm(img)), target))
    # create one-hot targets
    fmnist = fmnist.map(
        lambda img, target: (img, tf.one_hot(target, depth=10))
    )
    # cache this progress in memory, as there is no need to redo it; it
    # is deterministic after all
    fmnist = fmnist.cache()
    # shuffle, batch, prefetch
    fmnist = fmnist.shuffle(1000)
    fmnist = fmnist.batch(8)
    fmnist = fmnist.prefetch(20)
    # return preprocessed dataset
    return fmnist


train_ds = ds_train_fmnist.apply(prepare_fmnist_data)
test_ds = ds_test_fmnist.apply(prepare_fmnist_data)


# -------- task 2 "Model" ------------


def create_convlayer(
    filter_num=64, ker_size=3, stride=(1, 1), pad="same"
) -> tf.keras.layers.Conv2D:
    return tf.keras.layers.Conv2D(
        filters=filter_num, kernel_size=ker_size, strides=stride, padding=pad
    )


convlayer_1: tf.keras.layers.Conv2D = create_convlayer()
convlayer_2: tf.keras.layers.Conv2D = create_convlayer()


# ---------- task 4 "Visualization" ------------
_, ax = plt.subplot(2, 1, sharex=True, figsize=(9, 6))

ax[0].plot(train_losses, label="train loss")
ax[0].plot(test_losses, label="test loss")
ax[0].set(ylabel="Loss", title="Loss")
ax[0].legend(loc="upper right")

ax[1].plot(train_accuracies, label="train accuracy")
ax[1].plot(test_accuracies, label="test accuracy")
ax[1].set(ylabel="Accuracy", title="Accuracy")
ax[1].legend(loc="upper right")

plt.xlabel("Epochs")
plt.tight_layout()
plt.show()


input_shape = (16, 16)
n_outputs = 128

# create input of shape (1,16,16,1)
input_img = tf.random.uniform(shape=(1, input_shape[0], input_shape[1], 1))

# instantiate Conv2D layer with 128 filters with kernel size (16,16),
# without extra padding
convlayer = tf.keras.layers.Conv2D(
    filters=n_outputs, kernel_size=input_shape, strides=(1, 1), padding="valid"
)
# instantiate dense layer with 128 outputs
denselayer = tf.keras.layers.Dense(n_outputs)

# flatten input to process it with dense layer
flatted_input = tf.keras.layers.Flatten()(input_img)
denseoutput = denselayer(flatted_input)

# process input with convlayer
convoutput = convlayer(input_img)

# reshape weights from dense layer into shape of conv layer weights such
# that we can use the same weights for both
dense_weights = tf.reshape(
    denselayer.weights[0], shape=convlayer.weights[0].shape
)

dense_bias = tf.reshape(
    denselayer.weights[1], shape=convlayer.weights[1].shape
)

# assign weights from dense layer to conv layer to show they result in
# the same output
convlayer.weights[0].assign(dense_weights)
convlayer.weights[1].assign(dense_bias)

convoutput = convlayer(input_img)

convoutput = tf.reshape(convoutput, shape=denseoutput.shape)
