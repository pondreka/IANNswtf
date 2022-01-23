import tensorflow as tf
import re
import tensorflow_text as tf_txt
import math


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
    data = re.sub("[^A-Za-z ]+", "", data)
    # split text into words
    data_tokens = tf_txt.WhitespaceTokenizer().split(data)
    # TODO: change size to 1000
    bible = data_tokens[:100]

    bible_pairs = []
    context_window = 4
    sub_context_window = math.floor(context_window / 2)

    # iterate all bible finding the context words.
    for index, target_word in enumerate(bible):
        context_words = bible[
                        max(index - sub_context_window, 0): min(
                            index + sub_context_window + 1, len(bible)
                        )
                        ]

        for context_word in context_words:
            if context_word != target_word:
                # pair current target word with window word.
                bible_pairs.append((context_word, target_word))

    # tf.print(bible_pairs)

    return bible_pairs


def preprocess_dataset(ds):
    """Build a tensorflow-dataset from the original data.

    Args:
      ds (tensorflow_dataset): A dataset.

    Returns:
      prepared dataset
    """

    ds = ds.shuffle(256)
    ds = ds.batch(64)
    ds = ds.prefetch(128)

    return ds
