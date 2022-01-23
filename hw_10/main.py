import tensorflow as tf
import matplotlib.pyplot as plt
import os

from data_preparation import read_file, prepare_data, preprocess_dataset

os.environ["KMP_DUPLICATE_LIB_OK"] = "True"


def main():

    # -------- Task 1 "Data set" ---------
    bible = read_file("bible.txt")
    # -------- Task 2 "Word Embedding" ---------
    # -------- Task 2.1 "Preprocessing" ---------
    bible_pairs = prepare_data(bible)
    dataset = tf.data.Dataset.from_tensor_slices(bible_pairs)
    dataset = preprocess_dataset(dataset)


if __name__ == "__main__":
    main()
