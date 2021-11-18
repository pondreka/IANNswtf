
import tensorflow as tf
from custom_model import CustomModel
from dropout_model import DropoutModel
from hw_04.data_preparation import dataset_generation, prepare_dataframe, normalized_dataset_generation
from training_and_test import test, train_step
import matplotlib.pyplot as plt
import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "True"


def main():
    # ------------ task 1 "Data set" -------------
    train_df, valid_df, test_df = prepare_dataframe(csv_name="winequality-red.csv")

    train_ds = dataset_generation(train_df)
    train_ds_2 = normalized_dataset_generation(train_df)
    valid_ds = dataset_generation(valid_df)
    valid_ds_2 = normalized_dataset_generation(valid_df)
    test_ds = dataset_generation(test_df)
    test_ds_2 = normalized_dataset_generation(test_df)

    # --------- task 2 "Model" ----------
    model = CustomModel()
    dropout_model = DropoutModel()

    # --------- task 3 "Training" ---------
    num_epochs = 10
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

    # testing train dataset once before we starting the training
    train_loss, train_accuracy = test(model, train_ds, binary_loss)
    train_losses.append(train_loss)
    train_accuracies.append(train_accuracy)

    # testing valid dataset once before we starting the training
    valid_loss, valid_accuracy = test(model, valid_ds, binary_loss)
    valid_losses.append(valid_loss)
    valid_accuracies.append(valid_accuracy)

    # testing test dataset once before we starting the training
    test_loss, test_accuracy = test(model, test_ds, binary_loss)
    test_losses.append(test_loss)
    test_accuracies.append(test_accuracy)

    # We train for num_epochs epochs.
    for epoch in range(num_epochs):
        print(
            f"\nEpoch {epoch}: Train accuracy:\t{train_accuracies[-1]}"
            f"\n\t\t Valid accuracy:\t{valid_accuracies[-1]}"
            f"\n\t\t Test accuracy:\t\t{test_accuracies[-1]}"
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


    train_loss, train_accuracy = test(model, train_ds_2, binary_loss)
    train_losses_2.append(train_loss)
    train_accuracies_2.append(train_accuracy)

    valid_loss, valid_accuracy = test(model, valid_ds_2, binary_loss)
    valid_losses_2.append(valid_loss)
    valid_accuracies_2.append(valid_accuracy)

    # testing all data once before we begin
    test_loss, test_accuracy = test(model, test_ds_2, binary_loss)
    test_losses_2.append(test_loss)
    test_accuracies_2.append(test_accuracy)



    # We train for num_epochs epochs.
    for epoch in range(num_epochs):
        print(
            f"\nEpoch {epoch}: Train accuracy:\t{train_accuracies_2[-1]}"
            f"\n\t\t Valid accuracy:\t{valid_accuracies_2[-1]}"
            f"\n\t\t Test accuracy:\t\t{test_accuracies_2[-1]}"
        )

        # training (and checking in with training)
        epoch_loss_agg = []
        epoch_accuracy_agg = []
        for input, target in train_ds_2:
            train_loss, train_accuracy = train_step(
                dropout_model, input, target, binary_loss, adam_optimizer
            )
            epoch_loss_agg.append(train_loss)
            epoch_accuracy_agg.append(train_accuracy)

        # track training loss
        train_losses_2.append(tf.reduce_mean(epoch_loss_agg))
        train_accuracies_2.append(tf.reduce_mean(epoch_accuracy_agg))

        # testing, so we can track valid accuracy and valid loss
        valid_loss, valid_accuracy = test(dropout_model, valid_ds_2, binary_loss)
        valid_losses_2.append(valid_loss)
        valid_accuracies_2.append(valid_accuracy)

        test_loss, test_accuracy = test(dropout_model, test_ds_2, binary_loss)
        test_losses_2.append(test_loss)
        test_accuracies_2.append(test_accuracy)

    # ------------ task 5 "Visualization" ---------------
    # Visualize accuracy and loss for training and test data.

    fig, axes = plt.subplots(nrows=4, ncols=1, sharex=True, figsize=(9, 6))

    axes[0].plot(train_losses, label="train loss")
    axes[2].plot(train_accuracies, label="train accuracy")
    axes[0].plot(valid_losses, label="valid loss")
    axes[2].plot(valid_accuracies, label="valid accuracy")
    axes[0].plot(test_losses, label="test loss")
    axes[2].plot(test_accuracies, label="test accuracy")
    axes[0].set(ylabel="Loss", title="Loss Before Refinement")
    axes[0].legend(loc="upper right")
    axes[2].set(ylabel="Accuracy", title="Accuracy Before Refinement")
    axes[2].legend(loc="upper right")

    axes[1].plot(train_losses_2, label="train loss")
    axes[3].plot(train_accuracies_2, label="train accuracy")
    axes[1].plot(valid_losses_2, label="valid loss")
    axes[3].plot(valid_accuracies_2, label="valid accuracy")
    axes[1].plot(test_losses_2, label="test loss")
    axes[3].plot(test_accuracies_2, label="test accuracy")
    axes[1].set(ylabel="Loss", title="Loss After Refinement")
    axes[1].legend(loc="upper right")
    axes[3].set(ylabel="Accuracy", title="Accuracy After Refinement")
    axes[3].legend(loc="upper right")

    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    main()
