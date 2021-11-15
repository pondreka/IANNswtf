import tensorflow as tf
import pandas as pd
import numpy as np
from custom_model import CustomModel
from dropout_model import DropoutModel
from training_and_test import test, train_step
import matplotlib.pyplot as plt
import os

os.environ["KMP_DUPLICATE_LIB_OK"] = "True"


if __name__ == "__main__":

    # ------------ task 1 "Data set" -------------
    # ------------ task 1.1 "Load Data into a Dataframe" ---------------
    wine_df = pd.read_csv("winequality-red.csv", sep=";")

    # ------------ task 1.2 "Create a Tensorflow Dataset and Dataset Pipeline" -----------

    random_stat = np.random.RandomState()
    train_df = wine_df.sample(frac=0.7, random_state=random_stat)
    wine_df = wine_df.drop(train_df.index, axis=0)
    valid_df = wine_df.sample(frac=0.5, random_state=random_stat)
    test_df = wine_df.drop(valid_df.index, axis=0)

    def df_to_ds(df, batch_size=32):
        df = df.copy()
        labels = tf.squeeze(tf.constant([df.pop("quality")]), axis=0)

        ds = tf.data.Dataset.from_tensor_slices((df, labels))

        # create a binary target
        ds = ds.map(lambda input, target: (input, target > 5))

        ds = ds.cache()
        ds = ds.shuffle(128)
        ds = ds.batch(batch_size)
        ds = ds.prefetch(16)

        return ds

    train_ds = df_to_ds(train_df)
    valid_ds = df_to_ds(valid_df)
    test_ds = df_to_ds(test_df)

    # --------- task 2 "Model" ----------

    model = CustomModel()
    dropout_model = DropoutModel()

    # --------- task 3 "Training" ---------

    num_epochs = 100
    learning_rate = 0.001

    binary_loss = tf.keras.losses.BinaryCrossentropy()
    optimizer = tf.keras.optimizers.SGD(learning_rate)

    # Prepare some data for the final visualization
    # lists of tensors
    train_losses: list = []
    train_accuracies: list = []

    valid_losses: list = []
    valid_accuracies: list = []

    test_losses: list = []
    test_accuracies: list = []

    # testing all data once before we begin
    test_loss, test_accuracy = test(model, test_ds, binary_loss)
    test_losses.append(test_loss)
    test_accuracies.append(test_accuracy)

    valid_loss, valid_accuracy = test(model, valid_ds, binary_loss)
    valid_losses.append(valid_loss)
    valid_accuracies.append(test_accuracy)

    train_loss, train_accuracy = test(model, train_ds, binary_loss)
    train_losses.append(train_loss)
    train_accuracies.append(train_accuracy)

    # We train for num_epochs epochs.
    for epoch in range(num_epochs):
        print(
            f"Epoch {epoch}:\tTrain accuracy:\t{train_accuracies[-1]}"
            f"\n\t\tValid accuracy:\t{test_accuracies[-1]}"
        )

        # training (and checking in with training)
        epoch_loss_agg = []
        epoch_accuracy_agg = []
        for input, target in train_ds:
            train_loss, train_accuracy = train_step(
                model, input, target, binary_loss, optimizer
            )
            epoch_loss_agg.append(train_loss)
            epoch_accuracy_agg.append(train_accuracy)

        # track training loss
        train_losses.append(tf.reduce_mean(epoch_loss_agg))
        train_accuracies.append(tf.reduce_mean(epoch_accuracy_agg))

        # testing, so we can track valid accuracy and valid loss
        valid_loss, valid_accuracy = test(model, valid_ds, binary_loss)
        valid_losses.append(valid_loss)
        valid_accuracies.append(valid_accuracy)

        test_loss, test_accuracy = test(model, test_ds, binary_loss)
        test_losses.append(test_loss)
        test_accuracies.append(test_accuracy)

    # ------------ task 4 "Fine-Tuning" --------------
    adam_optimizer = tf.keras.optimizers.Adam(learning_rate)

    # Prepare some data for the final visualization
    # lists of tensors
    train_losses_2: list = []
    train_accuracies_2: list = []

    valid_losses_2: list = []
    valid_accuracies_2: list = []

    test_losses_2: list = []
    test_accuracies_2: list = []

    # testing all data once before we begin
    test_loss, test_accuracy = test(model, test_ds, binary_loss)
    test_losses_2.append(test_loss)
    test_accuracies_2.append(test_accuracy)

    valid_loss, valid_accuracy = test(model, valid_ds, binary_loss)
    valid_losses_2.append(valid_loss)
    valid_accuracies_2.append(test_accuracy)

    train_loss, train_accuracy = test(model, train_ds, binary_loss)
    train_losses_2.append(train_loss)
    train_accuracies_2.append(train_accuracy)

    # We train for num_epochs epochs.
    for epoch in range(num_epochs):
        print(
            f"Epoch {epoch}:\tTrain accuracy:\t{train_accuracies_2[-1]}"
            f"\n\t\tValid accuracy:\t{test_accuracies_2[-1]}"
        )

        # training (and checking in with training)
        epoch_loss_agg = []
        epoch_accuracy_agg = []
        for input, target in train_ds:
            train_loss, train_accuracy = train_step(
                dropout_model, input, target, binary_loss, adam_optimizer
            )
            epoch_loss_agg.append(train_loss)
            epoch_accuracy_agg.append(train_accuracy)

        # track training loss
        train_losses_2.append(tf.reduce_mean(epoch_loss_agg))
        train_accuracies_2.append(tf.reduce_mean(epoch_accuracy_agg))

        # testing, so we can track valid accuracy and valid loss
        valid_loss, valid_accuracy = test(dropout_model, valid_ds, binary_loss)
        valid_losses_2.append(valid_loss)
        valid_accuracies_2.append(valid_accuracy)

        test_loss, test_accuracy = test(dropout_model, test_ds, binary_loss)
        test_losses_2.append(test_loss)
        test_accuracies_2.append(test_accuracy)

    # ------------ task 5 "Visualization" ---------------
    # Visualize accuracy and loss for training and test data.

    fig, axes = plt.subplots(nrows=2, ncols=1, sharex=True, figsize=(9, 6))

    axes[0].plot(train_losses, label="train loss")
    axes[0].plot(train_accuracies, label="train accuracy")
    axes[0].plot(valid_losses, label="valid loss")
    axes[0].plot(valid_accuracies, label="valid accuracy")
    axes[0].plot(test_losses, label="test loss")
    axes[0].plot(test_accuracies, label="test accuracy")
    axes[0].set(ylabel="Loss/Accuracy", title="Before Refinement")
    axes[0].legend(loc="upper right")

    axes[1].plot(train_losses_2, label="train loss")
    axes[1].plot(train_accuracies_2, label="train accuracy")
    axes[1].plot(valid_losses_2, label="valid loss")
    axes[1].plot(valid_accuracies_2, label="valid accuracy")
    axes[1].plot(test_losses_2, label="test loss")
    axes[1].plot(test_accuracies_2, label="test accuracy")
    axes[1].set(ylabel="Loss/Accuracy", title="After Refinement")
    axes[1].legend(loc="upper right")

    plt.show()
