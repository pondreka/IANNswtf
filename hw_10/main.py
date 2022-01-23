import tensorflow as tf
import matplotlib.pyplot as plt
import os

from data_preparation import read_file, prepare_data, preprocess_dataset
from train_and_test import train_step, test
from skipgram import SkipGram

os.environ["KMP_DUPLICATE_LIB_OK"] = "True"


def main():

    # -------- Task 1 "Data set" ---------
    bible = read_file("bible.txt")
    # -------- Task 2 "Word Embedding" ---------
    # -------- Task 2.1 "Preprocessing" ---------
    bible_pairs = prepare_data(bible)
    dataset = tf.data.Dataset.from_tensor_slices(bible_pairs)
    train_data = dataset.take(7000)
    test_data = dataset.skip(7000)
    train_ds = preprocess_dataset(train_data)
    test_ds = preprocess_dataset(test_data)

    # -------- Task 2.2 "Model" ------------
    skipGram = SkipGram(10, 64)

    # -------- Task 2.3 "Training" -----------
    def visualization(train_losses, test_losses, name: str):
        plt.figure()
        (line1,) = plt.plot(train_losses)
        (line2,) = plt.plot(test_losses)
        plt.xlabel("Training steps")
        plt.ylabel(name)
        plt.legend((line1, line2), ("training", "test"))
        plt.show()

    num_epochs = 5
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

        visualization(train_losses, test_losses)

if __name__ == "__main__":
    main()
