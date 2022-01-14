import tensorflow as tf
import numpy as np


# --------- task 3 "Training" ---------------
def train_step(
    discriminator: tf.keras.Model,
    generator: tf.keras.Model,
    data: tf.Tensor
) -> (tf.Tensor, float):

    batch_size = 8
    random_input = tf.random.normal([batch_size, 100])

    with tf.GradientTape() as g_tape, tf.GradientTape() as d_tape:

        fake_data = generator(random_input, training=True)
        fake_data_pred = discriminator(fake_data, training=True)
        real_data_pred = discriminator(data, training=True)

        d_loss = discriminator.loss_function(real_data_pred, fake_data_pred) + tf.reduce_sum(discriminator.losses)
        g_loss = generator.loss_function(fake_data_pred) + tf.reduce_sum(generator.losses)

    d_gradients = d_tape.gradient(d_loss, discriminator.trainable_variables)
    discriminator.optimizer.apply_gradients(zip(d_gradients, discriminator.trainable_variables))

    g_gradients = g_tape.gradient(g_loss, generator.trainable_variables)
    generator.optimizer.apply_gradients(zip(g_gradients, generator.trainable_variables))

    discriminator.loss_metric.update_state(d_loss)
    generator.loss_metric.update_state(g_loss)

    return {discriminator.loss_metric.name: discriminator.loss_metric.result(),
            generator.loss_metric.name: generator.loss_metric.result()}


def test_step(
    discriminator: tf.keras.Model,
    generator: tf.keras.Model,
    data: tf.Tensor,
) -> (tf.Tensor, float):

    batch_size = 8
    random_input = tf.random.normal([batch_size, 100])

    fake_data = generator(random_input)
    fake_data_pred = discriminator(fake_data)
    real_data_pred = discriminator(data)

    d_loss = discriminator.loss_function(real_data_pred, fake_data_pred) + tf.reduce_sum(discriminator.losses)
    g_loss = generator.loss_function(fake_data_pred) + tf.reduce_sum(generator.losses)

    discriminator.loss_metric.update_state(d_loss)
    generator.loss_metric.update_state(g_loss)

    return {discriminator.loss_metric.name: discriminator.loss_metric.result(),
            generator.loss_metric.name: generator.loss_metric.result()}

