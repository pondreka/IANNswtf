import tensorflow as tf


# ---------- task 2 "Model" -----------
# ----------- task 2.1 "ResNet" -----------

class ResidualBlock(tf.keras.layers.Layer):

    def __init__(self):
        super(ResidualBlock, self).__init__()

        self.batch_norm_layer_1 = tf.keras.layers.BatchNormalization()
        self.activation_1 = tf.keras.layers.Activation(
            activation=tf.keras.activations.relu
        )

        self.conv_layer_1 = tf.keras.layers.Conv2D(
            filters=32,
            kernel_size=(1, 1),
            padding="same",
        )

        self.batch_norm_layer_2 = tf.keras.layers.BatchNormalization()
        self.activation_2 = tf.keras.layers.Activation(
            activation=tf.keras.activations.relu
        )

        self.conv_layer_2 = tf.keras.layers.Conv2D(
            filters=32,
            kernel_size=(3, 3),
            padding="same",
        )

        self.batch_norm_layer_3 = tf.keras.layers.BatchNormalization()
        self.activation_3 = tf.keras.layers.Activation(
            activation=tf.keras.activations.relu
        )

        self.conv_layer_3 = tf.keras.layers.Conv2D(
            filters=32,
            kernel_size=(1, 1),
            padding="same",
        )

    @tf.function
    def call(self, inputs: tf.Tensor, training: bool) -> tf.Tensor:
        out_batch_norm_1 = self.batch_norm_layer_1(inputs, training)
        out_activation_1 = self.activation_1(out_batch_norm_1)

        out_conv_layer_1 = self.conv_layer_1(out_activation_1)

        out_batch_norm_2 = self.batch_norm_layer_2(out_conv_layer_1, training)
        out_activation_2 = self.activation_2(out_batch_norm_2)

        out_conv_layer_2 = self.conv_layer_2(out_activation_2)

        out_batch_norm_3 = self.batch_norm_layer_3(out_conv_layer_2, training)
        out_activation_3 = self.activation_3(out_batch_norm_3)

        out_conv_layer_3 = self.conv_layer_3(out_activation_3)

        out = tf.keras.layers.Add()([inputs, out_conv_layer_3])

        return out


class ResNet(tf.keras.Model):

    def __init__(self, blocks: int = 3):
        super(ResNet, self).__init__()

        self.conv_layer = tf.keras.layers.Conv2D(
            filters=32,
            kernel_size=(1, 1),
            activation=tf.keras.activations.relu,
            input_shape=(32, 32, 3),
            padding="same",
        )

        self.res_blocks = [ResidualBlock() for _ in range(blocks)]

        self.pooling_layer = tf.keras.layers.GlobalAveragePooling2D()
        self.flatten_layer = tf.keras.layers.Flatten()
        self.dropout_layer = tf.keras.layers.Dropout(
            rate=0.5
        )
        self.output_layer = tf.keras.layers.Dense(
            10, activation=tf.keras.activations.softmax
        )

    @tf.function
    def call(self, inputs: tf.Tensor, training: bool = True) -> tf.Tensor:
        out = self.conv_layer(inputs)

        for res_block in self.res_blocks:
            out = res_block(out, training)

        out_pooling = self.pooling_layer(out)
        out_flatten = self.flatten_layer(out_pooling)
        out_dropout = self.dropout_layer(out_flatten, training)
        output = self.output_layer(out_dropout)

        return output


# ----------- task 2.2 "DenseNet" -----------
class TransitionLayer(tf.keras.layers.Layer):

    def __init__(self):
        super(TransitionLayer, self).__init__()
        self.batch_norm_layer_1 = tf.keras.layers.BatchNormalization()

        self.activation_1 = tf.keras.layers.Activation(
            activation=tf.keras.activations.relu
        )
        self.conv_layer_1 = tf.keras.layers.Conv2D(
            filters=8,
            strides=(2, 2),
            kernel_size=(1, 1),
            input_shape=(32, 32, 3),
            padding="valid"  # No padding
        )

        self.pooling_layer_1 = tf.keras.layers.AveragePooling2D(
            strides=(2, 2), padding="valid"
        )

    # TODO: Growth Rate?
    def call(self, inputs, training: bool):
        out = self.batch_norm_layer_1(inputs, training)
        out = self.activation_1(out)
        out = self.conv_layer_1(out)
        out = self.pooling_layer_1(out)
        return out


class DenseBlock(tf.keras.layers.Layer):

    def __init__(self):
        super(DenseBlock, self).__init__()

        self.batch_norm_layer_1 = tf.keras.layers.BatchNormalization()

        self.activation_1 = tf.keras.layers.Activation(
            activation=tf.keras.activations.relu
        )

        self.conv_layer_1 = tf.keras.layers.Conv2D(
            filters=8,
            kernel_size=(1, 1),
            activation=tf.keras.activations.relu,
            padding="valid",
        )

        self.batch_norm_layer_2 = tf.keras.layers.BatchNormalization()

        self.activation_2 = tf.keras.layers.Activation(
            activation=tf.keras.activations.relu
        )

        self.conv_layer_2 = tf.keras.layers.Conv2D(
            filters=8,
            kernel_size=(3, 3),
            activation=tf.keras.activations.relu,
            padding="same",
        )

    @tf.function
    def call(self, inputs: tf.Tensor, training: bool) -> tf.Tensor:
        out = self.batch_norm_layer_1(inputs, training)
        out = self.activation_1(out)
        out = self.conv_layer_1(out)
        out = self.batch_norm_layer_2(out, training)
        out = self.activation_2(out)
        out = self.conv_layer_2(out)
        out = tf.keras.layers.Concatenate()([inputs, out])

        return out


class DenseNet(tf.keras.Model):

    def __init__(self):
        super(DenseNet, self).__init__()

        self.conv_layer_1 = tf.keras.layers.Conv2D(
            filters=64,
            kernel_size=(3, 3),
            activation=tf.keras.activations.relu,
            input_shape=(32, 32, 3),
            padding="valid",
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
    def call(self, inputs: tf.Tensor, training: bool = False) -> tf.Tensor:
        out = self.conv_layer_1(inputs)
        out = self.dense_block_1(out, training)
        out = self.trans_layer_1(out, training)
        out = self.dense_block_2(out, training)
        out = self.trans_layer_2(out, training)
        out = self.dense_block_3(out, training)
        out = self.pooling_layer_1(out)
        out = self.flatten_layer(out)
        out = self.dropout_layer(out)
        out = self.output_layer(out)

        return out
