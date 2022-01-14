import tensorflow as tf


# ------------- task 2 "Model" -------------------
class Discriminator(tf.keras.Model):

    def __init__(self):
        super(Discriminator, self).__init__()

        learning_rate = 0.01
        self.optimizer = tf.keras.optimizers.Adam(learning_rate)

        self.fake_loss = tf.keras.losses.BinaryCrossentropy()
        self.real_loss = tf.keras.losses.BinaryCrossentropy()

        self.loss_metric = tf.keras.metrics.Mean(name="d_loss")

        l2_lambda = 0.01
        dropout_amount = 0.5

        self.all_layers = [
            tf.keras.layers.Conv2D(
                filters=32,
                kernel_size=5,
                strides=1,
                padding="same",
                activation=None,
                kernel_initializer=tf.keras.initializers.glorot_uniform,
                kernel_regularizer=tf.keras.regularizers.L2(l2_lambda)
            ),
            tf.keras.layers.BatchNormalization(),
            tf.keras.layers.Activation(tf.nn.relu),

            tf.keras.layers.MaxPool2D(pool_size=2, strides=1),
            tf.keras.layers.Dropout(dropout_amount),

            tf.keras.layers.Conv2D(
                filters=32,
                kernel_size=3,
                strides=1,
                padding="same",
                activation=None,
                kernel_initializer=tf.keras.initializers.glorot_uniform,
                kernel_regularizer=tf.keras.regularizers.L2(l2_lambda)
            ),
            tf.keras.layers.BatchNormalization(),
            tf.keras.layers.Activation(tf.nn.relu),

            tf.keras.layers.GlobalAveragePooling2D(),
            tf.keras.layers.BatchNormalization(),
            tf.keras.layers.Dropout(dropout_amount),

            tf.keras.layers.Dense(10, kernel_regularizer=tf.keras.regularizers.L2(l2_lambda)),
            tf.keras.layers.Activation(tf.nn.relu),

            tf.keras.layers.Dropout(dropout_amount),

            tf.keras.layers.Dense(1, kernel_regularizer=tf.keras.regularizers.L2(l2_lambda)),
            tf.keras.layers.Activation(tf.keras.activations.linear)

        ]

    @tf.function
    def call(self, x, training: bool = False):

        for layer in self.all_layers:
            try:
                x = layer(x, training)
            except:
                x = layer(x)

        return x

    def loss_function(self, real_data_pred, fake_data_pred):

        real_loss = self.real_loss(tf.ones_like(real_data_pred), real_data_pred)
        fake_loss = self.fake_loss(tf.zeros_like(fake_data_pred), fake_data_pred)

        loss = real_loss + fake_loss

        return -loss

    def reset_metrics(self):
        self.loss_metric.reset_states()
 

class Generator(tf.keras.Model):

    def __init__(self):
        super(Generator, self).__init__()

        learning_rate = 0.01
        self.optimizer = tf.keras.optimizers.Adam(learning_rate)

        self.fake_loss = tf.keras.losses.BinaryCrossentropy()

        self.loss_metric = tf.keras.metrics.Mean(name="g_loss")

        l2_lambda = 0.01
        dropout_amount = 0.5

        self.all_layers = [
            tf.keras.layers.Reshape((10, 10, 1), input_shape=(100,)),

            tf.keras.layers.Conv2DTranspose(
                filters=32,
                kernel_size=(6, 6),
                strides=(1, 1),
                padding="valid",
                activation=None,
                kernel_initializer=tf.keras.initializers.glorot_uniform,
                kernel_regularizer=tf.keras.regularizers.L2(l2_lambda)
            ),
            tf.keras.layers.BatchNormalization(),
            tf.keras.layers.Activation(tf.nn.relu),

            tf.keras.layers.Dropout(dropout_amount),

            tf.keras.layers.Conv2DTranspose(
                filters=32,
                kernel_size=(4, 4),
                strides=(1, 1),
                padding="valid",
                activation=None,
                kernel_initializer=tf.keras.initializers.glorot_uniform,
                kernel_regularizer=tf.keras.regularizers.L2(l2_lambda)
            ),
            tf.keras.layers.BatchNormalization(),
            tf.keras.layers.Activation(tf.nn.relu),

            tf.keras.layers.Dropout(dropout_amount),

            tf.keras.layers.Conv2DTranspose(
                filters=32,
                kernel_size=(2, 2),
                strides=(1, 1),
                padding="valid",
                activation=None,
                kernel_initializer=tf.keras.initializers.glorot_uniform,
                kernel_regularizer=tf.keras.regularizers.L2(l2_lambda)
            ),
            tf.keras.layers.BatchNormalization(),
            tf.keras.layers.Activation(tf.nn.relu),

            tf.keras.layers.Dropout(dropout_amount),

            tf.keras.layers.Conv2D(
                filters=1,
                kernel_size=(1, 1),
                strides=(1, 1),
                padding="valid",
                activation=None,
                kernel_initializer=tf.keras.initializers.glorot_uniform,
                kernel_regularizer=tf.keras.regularizers.L2(l2_lambda)
            ),
            tf.keras.layers.BatchNormalization(),
            tf.keras.layers.Activation(tf.nn.tanh)
        ]

    @tf.function
    def call(self, x, training: bool = False):

        for layer in self.all_layers:
            try:
                x = layer(x, training)
            except:
                x = layer(x)

        return x

    def loss_function(self, fake_data_pred):
        loss = self.fake_loss(tf.ones_like(fake_data_pred), fake_data_pred)

        return loss

    def reset_metrics(self):
        self.loss_metric.reset_states()
