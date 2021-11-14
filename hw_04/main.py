import pandas as pd
import tensorflow as tf
import numpy as np

# ---------- Task 1
# -------- Task 1.1

def prepare_ds(dataset, take_amount: int = 100000):
    """Prepares dataset with pipelines and returns specified amount"""
    # dataset = dataset.take(take_amount)
    dataset = dataset.map(
        lambda seq, label: (one_hot_converter(seq), tf.one_hot(label, 10))
    )
    dataset = dataset.cache()
    dataset = dataset.shuffle(1000)
    dataset = dataset.batch(32)
    dataset = dataset.prefetch(10)
    return dataset


def df_to_dataset(df, batch_size=32):
    df = df.copy()
    labels = tf.squeeze(tf.constant([df.pop("quality")]), axis=0)
    ds = tf.data.Dataset.from_tensor_slices(
        (dict(df), labels)
    ).batch(
        batch_size
    )
    return ds

wine_df = pd.read_csv("winequality-red.csv", sep=";")

# >>> wine_df.columns
# Index(['fixed acidity;"volatile acidity";"citric acid";"residual sugar";"chlorides";"free sulfur dioxide";"total sulfur dioxide";"density";"pH";"sulphates";"alcohol";"quality"'], dtype='object')
# make df into ds
full_size: int = len(wine_df)
train_size = int(0.7 * full_size)
valid_size = int(0.15 * full_size)

ran_stat = np.random.RandomState()
train_df = wine_df.sample(frac=0.7, random_state=ran_stat)
wine_df = wine_df.drop(train_df.index, axis=0)
valid_df = wine_df.sample(frac=0.5, random_state=ran_stat)
test_df = wine_df.drop(valid_df.index, axis=0)


labels = tf.squeeze([wine_df.pop("quality")], axis=0)
full_ds = tf.data.Dataset.from_tensor_slices((dict(wine_df), labels))
train_ds = full_ds.take(train_size)
remaining = full_ds.skip(train_size)
valid_ds = remaining.take(valid_size)
test_ds = remaining.skip(valid_size)
