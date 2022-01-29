import tensorflow as tf
import numpy as np

# --------- task 2.3 "Training" ---------------
# Definition of all the necessary functions for training


def train_step(
    model: tf.keras.Model,
    data: tf.Tensor,
    target: tf.Tensor,
    optimizer: tf.keras.optimizers,
) -> (tf.Tensor, float):
    """Training iteration over one input data.

    Args:
        model: Model to train.
        data: Data used to calculate the predictions.
        target: Targets for the loss and accuracy calculation.
        optimizer: an optimizer to apply with the gradient

    """
    with tf.GradientTape() as tape:
        loss = model(data, target)
    gradients = tape.gradient(loss, model.trainable_variables)
    optimizer.apply_gradients(zip(gradients, model.trainable_variables))

    return loss


def test(
    model: tf.keras.Model, test_data,
) -> (tf.Tensor, float):
    """Test iteration over all test data.

    Args:
        :param model: Model to train.
        :param test_data: Dataset to test with the model.

    """
    test_loss_aggregator = []
    for (data, target) in test_data:
        loss = model(data, target)
        test_loss_aggregator.append(loss.numpy())

    test_loss = tf.reduce_mean(test_loss_aggregator)

    # for word in test_words:
    #     embedding, _ = model(word)

    return test_loss
