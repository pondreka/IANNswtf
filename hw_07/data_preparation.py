import tensorflow as tf

# import pandas as pd
import numpy as np


# ------------------ Task 1.3 "Create a Data pipeline" --------------------
def prepare_data(ds):
    """Build a tensorflow-dataset from the original data.

    Args:
      ds (tensorflow_dataset): A dataset.

    Returns:
      prepared dataset
    """
    ds = ds.map(lambda seq, target: (seq, target))
    ds = ds.shuffle(256)
    ds = ds.batch(64)
    ds = ds.prefetch(128)

    return ds


# ------------- Task 1.2 "Generate a Tensorflow Dataset" --------------------


def integration_task(seq_len: int, num_samples: int):
    """ Generator of noise, target data
        Args:
            seq_len: number of noise signals
            num_samples: num of total samples
        Yields:
            tuple of signal and target
    """
    for _ in range(num_samples):
        signals = np.random.normal(0, 1, seq_len)
        # final integral (just positives)
        output = signals.sum(axis=-1) > 0
        output = output.astype(float)
        output = np.expand_dims(output, -1)
        signals = np.expand_dims(signals, -1)
        yield signals, output


def my_integration_task():
    """ Wrapper for the integration_task generator that defines the sequence length and number of samples """
    seq_len: int = 25
    num_samples: int = 10000

    while True:
        yield next(integration_task(seq_len, num_samples))


def create_dataset():
    """ Generate dataset from custom generators

        Return:
            Dataset
    """
    return tf.data.Dataset.from_generator(my_integration_task, output_types=(tf.float32, tf.float32))





