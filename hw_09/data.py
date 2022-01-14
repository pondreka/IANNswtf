import os
import matplotlib.pyplot as plt
import numpy as np
import urllib
import tensorflow as tf


# --------------task 1 "Data set" -----------------
def load_data():

    category = 'candle'

    if not os.path.isdir('npy_files'):
        os.mkdir('npy_files')

    url = f'https://storage.googleapis.com/quickdraw_dataset/full/numpy_bitmap/{category}.npy'
    urllib.request.urlretrieve(url, f'npy_files/{category}.npy')

    images = np.load(f'npy_files/{category}.npy')

    train_images = images[:10000]
    test_images = images[10000:14000]

    train_images = data_pipeline(train_images)
    test_images = data_pipeline(test_images)

    return train_images, test_images


def data_pipeline(data):

    ds = tf.data.Dataset.from_tensor_slices(data)
    ds = ds.map(lambda img: tf.reshape(img, (28, 28, 1)))
    ds = ds.map(lambda img: img / 128 - 1)

    ds = ds.cache()
    ds = ds.shuffle(1000)
    ds = ds.batch(8)
    ds = ds.prefetch(20)

    return ds


train, test = load_data()


