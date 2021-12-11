import tensorflow as tf
from train_and_test import train_step, test


def training(
    model, loss, num_epochs, optimizer, train_ds, valid_ds, test_ds,
):
    """Train the mode for the number of epochs specified.

    Args:
        model: model to train.
        loss: loss function used for the training and test the model.
        num_epochs: number of iterations for the training.
        optimizer:  Optimizer for the train step.
        train_ds:  training dataset
        valid_ds:  validation data set
        test_ds:  testing data set
        init_model:  model to use for the initial testing before the
            training iterations

    Returns:
        ((list, list), (list, list), (list, list)):
            3 touples, of 2-touples. Each of the three representing
            train, valid and valid datasets, for which the two inner
            ones represent their respective accuracy and loss.
    """

    # Prepare some data for the final visualization
    # lists of tensors
    train_losses: list = []
    train_accuracies: list = []
    valid_losses: list = []
    valid_accuracies: list = []
    test_losses: list = []
    test_accuracies: list = []
    # testing train dataset once before we starting the training
    train_loss, train_accuracy = test(model, train_ds, loss)
    train_losses.append(train_loss)
    train_accuracies.append(train_accuracy)
    # testing valid dataset once before we starting the training
    valid_loss, valid_accuracy = test(model, valid_ds, loss)
    valid_losses.append(valid_loss)
    valid_accuracies.append(valid_accuracy)
    # testing test dataset once before we starting the training
    test_loss, test_accuracy = test(model, test_ds, loss)
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
    """Prepares the visualization on 2 plots, for a single group.

    Args:
        axes: axis where to include the passed data to plot
        train_accuracies: all the collected training accuracies
        train_losses: all the collected training losses
        valid_accuracies: all the collected validation accuracies
        valid_losses: all the collected validation losses
        test_accuracies: all the collected testing accuracies
        test_losses: all the collected testing losses
        index: keeps track of the total plots to include outside of this
            specific preparation
        num_plot: total number of plots specified to do not try to plot
            nonexistent plots
        group_name: Name to be displayed on the top of the group plots.

    Returns:
        (axes, index): for the plot generated (in case they need
            adjusting)
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
