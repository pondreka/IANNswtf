import tensorflow as tf
import re
import tensorflow_text as tf_txt
import math
import numpy as np


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
    data = data.lower()
    data = data.replace("\n", " ")
    data = re.sub("[^A-Za-z ]+", "", data)
    # split text into words
    data_tokens = tf_txt.WhitespaceTokenizer().split(data)
    bible = data_tokens[:100000]

    tokens, ids, _ = map(lambda x: x.numpy(), tf.unique_with_counts(bible))

    word_to_id = {t.decode(): i for i, t in enumerate(tokens)}
    id_to_word = {word_to_id[x]: x for x in word_to_id}

    bible_pairs = []
    context_window = 4
    sub_context_window = math.floor(context_window / 2)

    # iterate all bible finding the context words.
    for index, target_word in enumerate(ids):
        context_words = ids[
                        max(index - sub_context_window, 0): min(
                            index + sub_context_window + 1, len(ids)
                        )
                        ]

        for context_word in context_words:
            if context_word != target_word:
                # pair current target word with window word.
                bible_pairs.append((context_word, target_word))

    return bible_pairs, id_to_word, word_to_id


def preprocess_dataset(ds):
    """Build a tensorflow-dataset from the original data.

    Args:
      ds (tensorflow_dataset): A dataset.

    Returns:
      prepared dataset
    """
    ds = ds.map(lambda inp: (inp[0], [inp[1],]))
    ds = ds.shuffle(256)
    ds = ds.batch(64)
    ds = ds.prefetch(20)

    return ds

