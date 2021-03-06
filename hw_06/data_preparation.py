import tensorflow as tf
import pandas as pd


def prepare_data(dataset):
    """Build a tensorflow-dataset from the original data.

    Args:
      dataset (tensorflow_dataset): A dataset.

    Returns:
      prepared dataset
    """
    # convert data from uint8 to float32
    ds = dataset.map(lambda img, target: (tf.cast(img, tf.float32), target))

    # sloppy input normalization, just bringing image values from range
    # [0, 255] to [0, 1]
    ds = ds.map(lambda img, target: ((img / tf.norm(img)), target))
    # create one-hot targets
    ds = ds.map(lambda img, target: (img, tf.one_hot(target, depth=10)))
    # cache this progress in memory, as there is no need to redo it; it
    # is deterministic after all
    ds = ds.cache()
    # shuffle, batch, prefetch
    ds = ds.shuffle(1000)
    ds = ds.batch(1024)
    ds = ds.prefetch(2048)
    # return preprocessed dataset
    return ds
