import tensorflow as tf


def prepare_f_mnist_data(f_mnist_dataset):
    """Build a tensorflow-dataset from the original data.

    Args:
      f_mnist_dataset (tensorflow_dataset): Fashion MNIST dataset.

    Returns:
      prepared dataset
    """
    # convert data from uint8 to float32
    f_mnist = f_mnist_dataset.map(
        lambda img, target: (tf.cast(img, tf.float32), target)
    )
    # sloppy input normalization, just bringing image values from range
    # [0, 255] to [-1, 1]
    f_mnist = f_mnist.map(lambda img, target: ((img / tf.norm(img)), target))
    # create one-hot targets
    f_mnist = f_mnist.map(
        lambda img, target: (img, tf.one_hot(target, depth=10))
    )
    # cache this progress in memory, as there is no need to redo it; it
    # is deterministic after all
    f_mnist = f_mnist.cache()
    # shuffle, batch, prefetch
    f_mnist = f_mnist.shuffle(1000)
    f_mnist = f_mnist.batch(8)
    f_mnist = f_mnist.prefetch(20)
    # return preprocessed dataset
    return f_mnist
