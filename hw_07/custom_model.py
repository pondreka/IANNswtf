import tensorflow as tf


# ---------- task 2 "Model" -----------
# ----------- task 2.1 "ResNet" -----------

class ResidualBlock(tf.keras.layers.Layer):
    """ Residual block layer definition """
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
    """ ResNet model definition

    Args:
        blocks: number of residual blocks in the model.
    """
    def __init__(self, blocks: int = 3):
        super(ResNet, self).__init__()

        self.conv_layer = tf.keras.layers.Conv2D(
            filters=32,
            kernel_size=(1, 1),
            input_shape=(32, 32, 3),
            padding="same",
        )

        self.res_blocks = [ResidualBlock() for _ in range(blocks)]

        self.batch_layer = tf.keras.layers.BatchNormalization()
        self.activation = tf.keras.layers.Activation(
            activation=tf.keras.activations.relu
        )

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

        out_batch = self.batch_layer(out, training)
        out_activation = self.activation(out_batch)
        out_pooling = self.pooling_layer(out_activation)
        out_flatten = self.flatten_layer(out_pooling)
        out_dropout = self.dropout_layer(out_flatten, training)
        output = self.output_layer(out_dropout)

        return output


# ----------- task 2.2 "DenseNet" -----------
class TransitionLayer(tf.keras.layers.Layer):
    """ Transition layer definition """
    def __init__(self):
        super(TransitionLayer, self).__init__()

        self.conv_layer = tf.keras.layers.Conv2D(
            filters=8,
            kernel_size=(1, 1),
            padding="valid"  # No padding
        )

        self.batch_norm_layer = tf.keras.layers.BatchNormalization()
        self.activation = tf.keras.layers.Activation(
            activation=tf.keras.activations.relu
        )

        self.pooling_layer = tf.keras.layers.AveragePooling2D(
            strides=(2, 2), padding="valid"
        )

    # TODO: Growth Rate?
    def call(self, inputs, training: bool):
        out_conv = self.conv_layer(inputs)
        out_batch_norm = self.batch_norm_layer(out_conv, training)
        out_activation = self.activation(out_batch_norm)
        out_pooling = self.pooling_layer(out_activation)
        return out_pooling


class Block(tf.keras.layers.Layer):
    """ Block layer definition """
    def __init__(self):
        super(Block, self).__init__()

        self.batch_norm_layer = tf.keras.layers.BatchNormalization()

        self.activation = tf.keras.layers.Activation(
            activation=tf.keras.activations.relu
        )

        self.conv_layer = tf.keras.layers.Conv2D(
            filters=8,
            kernel_size=(3, 3),
            padding="same",
        )

    @tf.function
    def call(self, inputs: tf.Tensor, training: bool) -> tf.Tensor:
        out_batch_norm = self.batch_norm_layer(inputs, training)
        out_activation = self.activation(out_batch_norm)
        out_conv = self.conv_layer(out_activation)

        return out_conv


class DenseBlock(tf.keras.layers.Layer):
    """ Dense block layer definition

    Args:
        blocks: number of blocks in the dense block layer.
    """
    def __init__(self, blocks: int = 2):
        super(DenseBlock, self).__init__()

        self.dense_blocks = [Block() for _ in range(blocks)]

        self.batch_norm_layer_1 = tf.keras.layers.BatchNormalization()

    @tf.function
    def call(self, inputs: tf.Tensor, training: bool) -> tf.Tensor:
        out = inputs

        for dense_block in self.dense_blocks:
            out = dense_block(out, training)

        out_concatenate = tf.keras.layers.Concatenate()([inputs, out])

        return out_concatenate


class DenseNet(tf.keras.Model):
    """ DenseNet model definition

    Args:
        blocks: number of dense blocks in the model.
    """
    def __init__(self, blocks: int = 2):
        super(DenseNet, self).__init__()

        self.conv_layer_1 = tf.keras.layers.Conv2D(
            filters=64,
            kernel_size=(3, 3),
            input_shape=(32, 32, 3),
            strides=(2, 2),
            padding="same",
        )

        self.batch_layer_1 = tf.keras.layers.BatchNormalization()
        self.activation_1 = tf.keras.layers.Activation(
            activation=tf.keras.activations.relu
        )

        self.pooling_layer_1 = tf.keras.layers.MaxPooling2D(pool_size=(3, 3), strides=(2, 2), padding="same")

        self.conv_layer_2 = tf.keras.layers.Conv2D(
            filters=8,
            kernel_size=(1, 1),
            padding="same",
        )

        self.dense_block_1 = DenseBlock()

        self.dense_blocks = [DenseBlock() for _ in range(blocks)]           # change number of conv layers here
        self.trans_layers = [TransitionLayer() for _ in range(blocks)]

        self.batch_layer_2 = tf.keras.layers.BatchNormalization()
        self.activation_2 = tf.keras.layers.Activation(
            activation=tf.keras.activations.relu
        )

        self.pooling_layer_2 = tf.keras.layers.GlobalAveragePooling2D()

        self.flatten_layer = tf.keras.layers.Flatten()
        self.dropout_layer = tf.keras.layers.Dropout(rate=0.5)
        self.output_layer = tf.keras.layers.Dense(
            10, activation=tf.keras.activations.softmax
        )

    @tf.function
    def call(self, inputs: tf.Tensor, training: bool = True) -> tf.Tensor:
        out_conv_1 = self.conv_layer_1(inputs)
        out_batch_1 = self.batch_layer_1(out_conv_1, training)
        out_activation_1 = self.activation_1(out_batch_1)
        out_pooling_1 = self.pooling_layer_1(out_activation_1)
        out_conv_2 = self.conv_layer_2(out_pooling_1)

        out = self.dense_block_1(out_conv_2, training)

        for i in range(len(self.dense_blocks)):
            out = self.trans_layers[i](out, training)
            out = self.dense_blocks[i](out, training)

        out_batch_2 = self.batch_layer_2(out, training)
        out_activation_2 = self.activation_2(out_batch_2)

        out_pooling_2 = self.pooling_layer_2(out_activation_2)
        out_flatten = self.flatten_layer(out_pooling_2)
        out_dropout = self.dropout_layer(out_flatten, training)
        out_output = self.output_layer(out_dropout)

        return out_output
