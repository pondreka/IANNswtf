import tensorflow as tf


# ---------- task 2 "Model" -----------

# Cell


class LSTM_Cell(tf.keras.layers.Layer):
    # TODO: document

    def __init__(self, units):
        super(LSTM_Cell, self).__init__()

        self.units = units

        self.dense_forget_gate = tf.keras.layers.Dense(
            units,
            activation=tf.nn.sigmoid,
            bias_initializer=tf.keras.initializers.Zeros(),
        )

        self.dense_input_gate = tf.keras.layers.Dense(
            units, activation=tf.nn.sigmoid
        )

        self.dense_state_candidates = tf.keras.layers.Dense(
            units, activation=tf.nn.tanh
        )

        self.dense_output = tf.keras.layers.Dense(
            units, activation=tf.nn.sigmoid
        )

        # self.bias = tf.Variable(tf.zeros(units), name="LSTM_Cell_biases")

        self.state_size = units

    def call(self, inputs, state, training=False):
        # TODO: document

        # Preparing
        out_concatenate = tf.concat((state[0], inputs), axis=1)

        forget_gate = self.dense_forget_gate(out_concatenate)

        input_gate = self.dense_input_gate(out_concatenate)

        state_candidates = self.dense_input_gate(out_concatenate)

        # Update cell state
        state[1] = forget_gate * state[1] + input_gate * state_candidates

        # Determining the hidden state/output
        output_gate = self.dense_input_gate(out_concatenate)

        state[0] = output_gate * tf.math.tanh(state[1])

        return state


# Wrapper


class LSTM_Layer(tf.keras.layers.Layer):
    def __init__(self, cell_units, num_cells):
        super(LSTM_Layer, self).__init__()

        self.cells = [LSTM_Cell(cell_units) for _ in num_cells]

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
