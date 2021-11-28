import tensorflow as tf


# ---------- task 2 "Model" -----------
# ----------- task 2.1 "ResNet" -----------

class ResidualBlock(tf.keras.layers.Layer):

    def __init__(self):
        super(ResidualBlock, self).__init__()

        self.batch_norm_layer_1 = tf.keras.layers.BatchNormalization()

        self.conv_layer_1 = tf.keras.layers.Conv2D(
            filters=3,
            kernel_size=(1, 1),
            activation=tf.keras.activations.relu,
            input_shape=(32, 32, 3),
            padding="same",
        )

        self.batch_norm_layer_2 = tf.keras.layers.BatchNormalization()

        self.conv_layer_2 = tf.keras.layers.Conv2D(
            filters=3,
            kernel_size=(3, 3),
            activation=tf.keras.activations.relu,
            padding="same",
        )

        self.batch_norm_layer_3 = tf.keras.layers.BatchNormalization()

        self.conv_layer_3 = tf.keras.layers.Conv2D(
            filters=8,
            kernel_size=(1, 1),
            activation=tf.keras.activations.relu,
            padding="same",
        )

    @tf.function
    def call(self, inputs: tf.Tensor, test: bool) -> tf.Tensor:
        if test:
            out_1 = self.conv_layer_1(inputs)
            out_2 = self.conv_layer_2(out_1)
            out_3 = self.conv_layer_3(out_2)
            out = tf.keras.layers.Add()([out_3, out_1])
        else:
            out_1 = self.batch_norm_layer_1(inputs)
            out_2 = self.conv_layer_1(out_1)
            out_3 = self.batch_norm_layer_2(out_2)
            out_4 = self.conv_layer_2(out_3)
            out_5 = self.batch_norm_layer_3(out_4)
            out_6 = self.conv_layer_3(out_5)
            out = tf.keras.layers.Add()([out_6, out_1])

        return out


class ResNet(tf.keras.Model):

    def __init__(self):
        super(ResNet, self).__init__()

        self.conv_layer_1 = tf.keras.layers.Conv2D(
            filters=8,
            kernel_size=(1, 1),
            activation=tf.keras.activations.relu,
            input_shape=(32, 32, 3),
            padding="same",
        )

        self.res_block_1 = ResidualBlock()
        self.res_block_2 = ResidualBlock()
        self.res_block_3 = ResidualBlock()

        self.pooling_layer_1 = tf.keras.layers.GlobalAveragePooling2D()

        self.flatten_layer = tf.keras.layers.Flatten()
        self.dropout_layer = tf.keras.layers.Dropout(rate=0.5)
        self.output_layer = tf.keras.layers.Dense(
            10, activation=tf.keras.activations.softmax
        )

    @tf.function
    def call(self, inputs: tf.Tensor, test: bool = False) -> tf.Tensor:
        out_1 = self.conv_layer_1(inputs)
        out_2 = self.res_block_1(out_1, test)
        out_3 = self.res_block_2(out_2, test)
        out_4 = self.res_block_3(out_3, test)
        out_5 = self.pooling_layer_1(out_4)
        out_6 = self.flatten_layer(out_5)
        out_7 = self.dropout_layer(out_6)
        out_8 = self.output_layer(out_7)

        return out_8


# ----------- task 2.2 "DenseNet" -----------
class TransitionLayer(tf.keras.layers.Layer):

    def __init__(self):
        super(TransitionLayer, self).__init__()

        self.conv_layer_1 = tf.keras.layers.Conv2D(
            filters=8,
            kernel_size=(1, 1),
            activation=tf.keras.activations.relu,
            input_shape=(32, 32, 3),
            padding="same",
        )

        self.batch_norm_layer_1 = tf.keras.layers.BatchNormalization()

        self.pooling_layer_1 = tf.keras.layers.AveragePooling2D(
            strides=(1, 1), padding="same"
        )

    # TODO: Growth Rate?
    def call(self, inputs, test: bool):
        out = self.conv_layer_1(inputs)
        if not test:
            out = self.batch_norm_layer_1(out)
        out_2 = self.pooling_layer_1(out)
        return out_2


class DenseBlock(tf.keras.layers.Layer):

    def __init__(self):
        super(DenseBlock, self).__init__()

        self.batch_norm_layer_1 = tf.keras.layers.BatchNormalization()

        self.conv_layer_1 = tf.keras.layers.Conv2D(
            filters=8,
            kernel_size=(1, 1),
            activation=tf.keras.activations.relu,
            padding="same",
        )

        self.batch_norm_layer_2 = tf.keras.layers.BatchNormalization()

        self.conv_layer_2 = tf.keras.layers.Conv2D(
            filters=8,
            kernel_size=(3, 3),
            activation=tf.keras.activations.relu,
            padding="same",
        )

        self.batch_norm_layer_3 = tf.keras.layers.BatchNormalization()

        self.conv_layer_3 = tf.keras.layers.Conv2D(
            filters=3,
            kernel_size=(1, 1),
            activation=tf.keras.activations.relu,
            padding="same",
        )

    @tf.function
    def call(self, inputs: tf.Tensor, test: bool) -> tf.Tensor:
        if test:
            out_1 = self.conv_layer_1(inputs)
            out_2 = self.conv_layer_2(out_1)
            out_3 = self.conv_layer_3(out_2)
            out = tf.keras.layers.Concatenate()([out_3, out_1])
        else:
            out_1 = self.batch_norm_layer_1(inputs)
            out_2 = self.conv_layer_1(out_1)
            out_3 = self.batch_norm_layer_2(out_2)
            out_4 = self.conv_layer_2(out_3)
            out_5 = self.batch_norm_layer_3(out_4)
            out_6 = self.conv_layer_3(out_5)
            out = tf.keras.layers.Concatenate()([out_6, out_1])

        return out


class DenseNet(tf.keras.Model):

    def __init__(self):
        super(DenseNet, self).__init__()

        self.conv_layer_1 = tf.keras.layers.Conv2D(
            filters=8,
            kernel_size=(1, 1),
            activation=tf.keras.activations.relu,
            input_shape=(32, 32, 3),
            padding="same",
        )

        self.dense_block_1 = DenseBlock()
        self.trans_layer_1 = TransitionLayer()
        self.dense_block_2 = DenseBlock()
        self.trans_layer_2 = TransitionLayer()
        self.dense_block_3 = DenseBlock()

        self.pooling_layer_1 = tf.keras.layers.GlobalAveragePooling2D()

        self.flatten_layer = tf.keras.layers.Flatten()
        self.dropout_layer = tf.keras.layers.Dropout(rate=0.5)
        self.output_layer = tf.keras.layers.Dense(
            10, activation=tf.keras.activations.softmax
        )

    @tf.function
    def call(self, inputs: tf.Tensor, test: bool = False) -> tf.Tensor:
        out_1 = self.conv_layer_1(inputs)
        out_2 = self.dense_block_1(out_1, test)
        out_2_1 = self.trans_layer_1(out_2, test)
        out_3 = self.dense_block_2(out_2_1, test)
        out_3_1 = self.trans_layer_2(out_3, test)
        out_4 = self.dense_block_3(out_3_1, test)
        out_5 = self.pooling_layer_1(out_4)
        out_6 = self.flatten_layer(out_5)
        out_7 = self.dropout_layer(out_6)
        out_8 = self.output_layer(out_7)

        return out_8
