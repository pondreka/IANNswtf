import tensorflow as tf
from custom_model import CustomModel
from dropout_model import DropoutModel
from data_preparation import (
    dataset_generation,
    prepare_dataframe,
    normalized_dataset_generation,
)
from training_and_test import test, train_step
import matplotlib.pyplot as plt
import os

os.environ["KMP_DUPLICATE_LIB_OK"] = "True"


def main():
    # ------------ task 1 "Data set" -------------
    train_df, valid_df, test_df = prepare_dataframe(
        csv_name="winequality-red.csv"
    )

    batch_size = 64

    train_ds = dataset_generation(train_df, batch_size=batch_size)
    valid_ds = dataset_generation(valid_df, batch_size=batch_size)
    test_ds = dataset_generation(test_df, batch_size=batch_size)

    train_ds_norm = normalized_dataset_generation(
        train_df, batch_size=batch_size
    )
    valid_ds_norm = normalized_dataset_generation(
        valid_df, batch_size=batch_size
    )
    test_ds_norm = normalized_dataset_generation(
        test_df, batch_size=batch_size
    )

    # --------- task 2 "Model" ----------
    model = CustomModel()
    dropout_rate = 0.8
    dropout_model = DropoutModel(rate=dropout_rate)
    dropout_rate_norm = 0.8
    dropout_model_norm = DropoutModel(rate=dropout_rate_norm)

    # --------- task 3 "Training" ---------
    num_epochs = 100
    learning_rate = 0.0025

    binary_loss = tf.keras.losses.BinaryCrossentropy()
    sgd_optimizer = tf.keras.optimizers.SGD(learning_rate)

    train_pre, valid_pre, test_pre = training(
        model=model,
        loss=binary_loss,
        num_epochs=num_epochs,
        optimizer=sgd_optimizer,
        train_ds=train_ds,
        valid_ds=valid_ds,
        test_ds=test_ds,
    )

    # ------------ task 4 "Fine-Tuning" --------------
    # Train 2 models for comparison
    adam_optimizer = tf.keras.optimizers.Adam(learning_rate)

    train, valid, test = training(
        model=dropout_model,
        loss=binary_loss,
        num_epochs=num_epochs,
        optimizer=adam_optimizer,
        train_ds=train_ds,
        valid_ds=valid_ds,
        test_ds=test_ds,
    )

    train_norm, valid_norm, test_norm = training(
        model=dropout_model_norm,
        loss=binary_loss,
        num_epochs=num_epochs,
        optimizer=adam_optimizer,
        train_ds=train_ds_norm,
        valid_ds=valid_ds_norm,
        test_ds=test_ds_norm,
    )

    # ------------ task 5 "Visualization" ---------------
    # Visualize accuracy and loss for training and test data.

    num_plot_visualization = 3
    _, axes = plt.subplots(
        nrows=num_plot_visualization * 2, ncols=1, sharex=True, figsize=(9, 6)
    )

    accuracies = 0
    losses = 1

    index = 0

    axes, index = prepare_visualization(
        axes,
        train_pre[accuracies],
        train_pre[losses],
        valid_pre[accuracies],
        valid_pre[losses],
        test_pre[accuracies],
        test_pre[losses],
        index,
        num_plot_visualization,
        group_name=f"Pre-optimized: epochs={num_epochs}, lr={learning_rate}, "
        f"batch={batch_size}",
    )

    axes, index = prepare_visualization(
        axes,
        train[accuracies],
        train[losses],
        valid[accuracies],
        valid[losses],
        test[accuracies],
        test[losses],
        index,
        num_plot_visualization,
        group_name=f"Optimized: epochs={num_epochs}, lr={learning_rate}, "
        f"batch={batch_size}, dropout_rate={dropout_rate}",
    )

    axes, index = prepare_visualization(
        axes,
        train_norm[accuracies],
        train_norm[losses],
        valid_norm[accuracies],
        valid_norm[losses],
        test_norm[accuracies],
        test_norm[losses],
        index,
        num_plot_visualization,
        group_name=f"Optimized (norm): epochs={num_epochs}, lr={learning_rate}, "
        f"batch={batch_size}, dropout_rate={dropout_rate_norm}",
    )

    plt.tight_layout()
    plt.show()


def training(
    model,
    loss,
    num_epochs,
    optimizer,
    train_ds,
    valid_ds,
    test_ds,
    init_model=None,
):
    """Train the mode for the number of epochs specified.

    :param model: model to train.
    :param loss: loss function used for the training and test the model.
    :param num_epochs: number of iterations for the training.
    :param optimizer:  Optimizer for the train step.
    :param train_ds:  training dataset
    :param valid_ds:  validation data set
    :param test_ds:  testing data set
    :param init_model:  model to use for the initial testing before the training iterations
    """

    if init_model is None:
        init_model = model
    # Prepare some data for the final visualization
    # lists of tensors
    train_losses: list = []
    train_accuracies: list = []
    valid_losses: list = []
    valid_accuracies: list = []
    test_losses: list = []
    test_accuracies: list = []
    # testing train dataset once before we starting the training
    train_loss, train_accuracy = test(init_model, train_ds, loss)
    train_losses.append(train_loss)
    train_accuracies.append(train_accuracy)
    # testing valid dataset once before we starting the training
    valid_loss, valid_accuracy = test(init_model, valid_ds, loss)
    valid_losses.append(valid_loss)
    valid_accuracies.append(valid_accuracy)
    # testing test dataset once before we starting the training
    test_loss, test_accuracy = test(init_model, test_ds, loss)
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
        for inp, target in train_ds:
            train_loss, train_accuracy = train_step(
                model, inp, target, loss, optimizer
            )
            epoch_loss_agg.append(train_loss)
            epoch_accuracy_agg.append(train_accuracy)

        # track training loss
        train_losses.append(tf.reduce_mean(epoch_loss_agg))
        train_accuracies.append(tf.reduce_mean(epoch_accuracy_agg))

        # testing, so we can track valid accuracy and valid loss
        valid_loss, valid_accuracy = test(model, valid_ds, loss)
        valid_losses.append(valid_loss)
        valid_accuracies.append(valid_accuracy)

        test_loss, test_accuracy = test(model, test_ds, loss)
        test_losses.append(test_loss)
        test_accuracies.append(test_accuracy)

    return (
        (train_accuracies, train_losses),
        (valid_accuracies, valid_losses),
        (test_accuracies, test_losses),
    )


def prepare_visualization(
    axes,
    train_accuracies,
    train_losses,
    valid_accuracies,
    valid_losses,
    test_accuracies,
    test_losses,
    index,
    num_plot,
    group_name,
):
    """Prepares the visualization on 2 plots, loss and accuracy, for a single group.

    :param axes: axis where to include the passed data to plot
    :param train_accuracies: all the collected training accuracies
    :param train_losses: all the collected training losses
    :param valid_accuracies: all the collected validation accuracies
    :param valid_losses: all the collected validation losses
    :param test_accuracies: all the collected testing accuracies
    :param test_losses: all the collected testing losses
    :param index: keeps track of the total plots to include outside of this specific preparation
    :param num_plot: total number of plots specified to do not try to plot nonexistent plots
    :param group_name: Name to be displayed on the top of the group plots.
    """

    if index > num_plot * 2 - 1:
        print("Warning: Not enough plots per visualization defined!")
        return axes, index

    axes[index].plot(train_losses, label="train loss")
    axes[index].plot(valid_losses, label="valid loss")
    axes[index].plot(test_losses, label="test loss")
    axes[index].set(ylabel="Loss", title=f"{group_name}")
    axes[index].legend(loc="upper right")

    axes[index + 1].plot(train_accuracies, label="train accuracy")
    axes[index + 1].plot(valid_accuracies, label="valid accuracy")
    axes[index + 1].plot(test_accuracies, label="test accuracy")
    axes[index + 1].set(ylabel="Accuracy")
    axes[index + 1].legend(loc="upper right")
    index += 2
    return axes, index


if __name__ == "__main__":
    main()
