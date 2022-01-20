import tensorflow as tf
import matplotlib.pyplot as plt
import os
import math

from data_preparation import read_file, prepare_data, preprocess_dataset

os.environ["KMP_DUPLICATE_LIB_OK"] = "True"


def main():

    # -------- Task 1 "Data set" ---------
    bible = read_file("bible.txt")
    # -------- Task 2 "Word Embedding" ---------
    # -------- Task 2.1 "Preprocessing" ---------
    bible = prepare_data(bible)
    context_window = 4
    sub_context_window = math.floor(context_window / 2)

    # TODO: input-target pairs
    bible_pairs = []

    # iterate all bible finding the context words.
    for index, target_word in enumerate(bible):
        context_words = bible[
            max(index - sub_context_window, 0) : min(
                index + sub_context_window + 1, len(bible)
            )
        ]

        for context_word in context_words:
            if context_word != target_word:
                # pair current target word with window word.
                bible_pairs.append((context_word, target_word))

    tf.print(bible_pairs)


if __name__ == "__main__":
    main()
