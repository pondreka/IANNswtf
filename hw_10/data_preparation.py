import tensorflow as tf
import re
import tensorflow_text as tf_txt


def read_file(file_name):
    with open(file_name) as file:
        file_txt = file.read()

    return file_txt


def prepare_data(data):
    """ Prepare all the data as required: lowercase, remove newlines and special characters, and split into tokens.

    Args:
      data Data to prepare.

    Returns:
      prepared data
    """
    data.lower()
    data.replace("\n", "")
    data = re.sub('[^A-Za-z ]+', '', data)
    # split text into words
    data_tokens = tf_txt.WhitespaceTokenizer().split(data)
    data_tokens = data_tokens[:10000]
    return data_tokens


def preprocess_dataset(ds):
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
