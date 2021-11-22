import tensorflow as tf
import numpy as np

# --------- task 3 "Training" ---------------
# Definition of all the necessary functions for training


def train_step(
    model: tf.keras.Model,
    data: tf.Tensor,
    target: tf.Tensor,
    loss_function: tf.keras.losses,
    optimizer: tf.keras.optimizers,
) -> (tf.Tensor, float):
    """Training iteration over one input data.

    Args:
        model: Model to train.
        data: Data used to calculate the predictions.
        target: Targets for the loss and accuracy calculation.
        loss_function: Function used to calculate the loss.
        optimizer: an optimizer to apply with the gradient

    Returns:
        (tf.Tensor, tf.Tensor): A tuple with the calculated loss and
            accuracy.
    """

    with tf.GradientTape() as tape:
        prediction = model(data)
        loss = loss_function(target, prediction)
        accuracy = target == np.round(prediction)
        accuracy = np.mean(accuracy)
        gradients = tape.gradient(loss, model.trainable_variables)
    optimizer.apply_gradients(zip(gradients, model.trainable_variables))

    return loss, accuracy


def test(
    model: tf.keras.Model, test_data, loss_function: tf.keras.losses
) -> (tf.Tensor, float):
    """Test iteration over all test data.

    Args:
        :param model: Model to train.
        :param test_data: Dataset to test with the model.
        :param loss_function: Function used to calculate the loss.

    Returns:
        (float, float): A tuple with the calculated loss and accuracy
    """
    test_accuracy_aggregator = []
    test_loss_aggregator = []
    for (data, target) in test_data:
        prediction = model(data)
        sample_test_loss = loss_function(target, prediction)
        sample_test_accuracy = target == np.round(prediction)
        sample_test_accuracy = np.mean(sample_test_accuracy)
        test_loss_aggregator.append(sample_test_loss.numpy())
        test_accuracy_aggregator.append(np.mean(sample_test_accuracy))

    test_loss = tf.reduce_mean(test_loss_aggregator)
    test_accuracy = tf.reduce_mean(test_accuracy_aggregator)

    return test_loss, test_accuracy
