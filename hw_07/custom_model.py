import tensorflow as tf


# ---------- task 2 "Model" -----------

# Wrapper


class RNNWrapper(tf.keras.layers.Layer):
    def __init__(self, RNN_Cell, return_sequences=False):
        super(RNNWrapper, self).__init__()

        self.return_sequences = return_sequences

        self.cell = RNN_Cell

    def call(self, data, training=False):

        length = data.shape[1]

        # initialize state of the simple rnn cell
        state = tf.zeros((data.shape[0], self.cell.units), tf.float32)

        # initialize array for hidden states (only relevant if self.return_sequences == True)
        hidden_states = tf.TensorArray(dtype=tf.float32, size=length)

        for t in tf.range(length):
            input_t = data[:, t, :]

            state = self.cell(input_t, state, training)

            if self.return_sequences:
                # write the states to the TensorArray
                # hidden_states = hidden_states.write(t, state)
                hidden_states.append(state)

        if self.return_sequences:
            # transpose the sequence of hidden_states from TensorArray accordingly
            # (batch and time dimensions are otherwise switched after .stack())
            outputs = tf.transpose(hidden_states.stack(), [1, 0, 2])

        else:
            # take the last hidden state of the simple rnn cell
            outputs = state

        return outputs


# Cell


class CustomSimpleRNNCell(tf.keras.layers.Layer):
    def __init__(self, units, kernel_regularizer=None):
        super(CustomSimpleRNNCell, self).__init__()

        self.units = units

        self.dense_hstate = tf.keras.layers.Dense(
            units, kernel_regularizer=kernel_regularizer, use_bias=False
        )

        self.dense_input = tf.keras.layers.Dense(
            units, kernel_regularizer=kernel_regularizer, use_bias=False
        )

        self.bias = tf.Variable(tf.zeros(units), name="RNN_Cell_biases")

        self.state_size = units

    def call(self, input_t, state, training=False):
        # we compute the sum of the input at t matrix multiplied and the previous state matrix multiplied
        # and an additional bias added.
        x_sum = (
            self.dense_input(input_t) + self.dense_hstate(state) + self.bias
        )

        # finally we use hyperbolic tangent as an activation function to update the RNN cell state
        state = tf.nn.tanh(x_sum)

        return state


# Model


class RNN_Model(tf.keras.Model):
    def __init__(self, units):
        super(RNN_Model, self).__init__()

        self.RNNWrapper = RNNWrapper(
            CustomSimpleRNNCell(units), return_sequences=False
        )

        self.dense = tf.keras.layers.Dense(128, activation="relu")

        self.out = tf.keras.layers.Dense(10, activation="softmax")

    # @tf.function(experimental_relax_shapes=True)
    def call(self, data, training=False):
        x = self.RNNWrapper(data, training)
        x = self.dense(x)
        x = self.out(x)

        return x
