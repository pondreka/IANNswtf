
import pandas as pd
import numpy as np
import tensorflow as tf


def prepare_dataframe(csv_name: str) -> tuple:
    """

    """
    # ------------ task 1.1 "Load Data into a Dataframe" ---------------
    wine_df = pd.read_csv(csv_name, sep=";")

    # ------------ task 1.2 "Create a Tensorflow Dataset and Dataset Pipeline" -----------
    random_stat = np.random.RandomState()
    train_df = wine_df.sample(frac=0.7, random_state=random_stat)
    wine_df = wine_df.drop(train_df.index, axis=0)
    valid_df = wine_df.sample(frac=0.5, random_state=random_stat)
    test_df = wine_df.drop(valid_df.index, axis=0)

    return train_df, valid_df, test_df


def dataset_generation(dataframe, batch_size: int = 32):
    """
    Build a Tensorflow dataset from the dataframes and apply pipelines

    """
    df = dataframe.copy()
    labels = tf.squeeze(tf.constant([df.pop("quality")]), axis=0)
    # we normalize the input before adding it to ds
    # df = df / df.sum(axis=0)
    ds = tf.data.Dataset.from_tensor_slices((df, labels))

    # create a binary target
    ds = ds.map(lambda input, target: (input, target > 5))

    ds = ds.cache()
    ds = ds.shuffle(128)
    ds = ds.batch(batch_size)
    ds = ds.prefetch(16)

    return ds

