import tensorflow as tf
import os

from data_preparation import read_file, prepare_data, preprocess_dataset
from train_and_test import train_step, test
from skipgram import SkipGram
from visualization import visualization, print_closest_word

os.environ["KMP_DUPLICATE_LIB_OK"] = "True"


def main():

    # -------- Task 1 "Data set" ---------
    # path = tf.keras.utils.get file(”nietzsche.txt”,
    # origin=”https://s3.amazonaws.com/text-datasets/nietzsche.txt”)
    # TODO: pass link string to get_file
    text = get_file()

    # -------- Task 2 "Word Embedding" ---------
    # -------- Task 2.1 "Preprocessing" ---------
    bible_pairs, id_to_word, word_to_id = prepare_data(text)
    dataset = tf.data.Dataset.from_tensor_slices(bible_pairs)
    train_data = dataset.take(7000)
    test_data = dataset.skip(7000)
    train_ds = preprocess_dataset(train_data)
    test_ds = preprocess_dataset(test_data)

    test_words = ['god', 'day', 'moses', 'eden', 'lord', 'name']

    # -------- Task 2.2 "Model" ------------
    skipGram = SkipGram(len(id_to_word), 64)

    # -------- Task 2.3 "Training" -----------
    num_epochs = 15
    learning_rate = 0.01
    optimizer = tf.keras.optimizers.Adam(learning_rate)

    train_losses = []
    test_losses = []

    for epoch in range(num_epochs):
        print(f"Epoch {epoch}:")

        train_loss = []

        for (input, target) in train_ds:
            loss = train_step(skipGram, input, target, optimizer)
            train_loss.append(loss)

        loss_train = tf.reduce_mean(train_loss)
        print(f"train loss {loss_train}")

        train_losses.append(loss_train)

        test_loss = test(skipGram, test_ds)
        print(f"test loss {test_loss}")
        test_losses.append(test_loss)

        print_closest_word(skipGram, test_words, word_to_id, id_to_word)
        visualization(train_losses, test_losses)


if __name__ == "__main__":
    main()
