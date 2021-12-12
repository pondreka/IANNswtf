import tensorflow as tf
from data_preparation import create_dataset


# ---------- task 2 "The network" -----------

# Cell


class LSTM_Cell(tf.keras.layers.Layer):
    # TODO: document

    def __init__(self, units):
        super(LSTM_Cell, self).__init__()

        self.dense_forget_gate = tf.keras.layers.Dense(
            units,
            activation=tf.nn.sigmoid,
            bias_initializer=tf.keras.initializers.Ones(),
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

        self.state_size = units

    @tf.function
    def call(self, inputs, hidden_state, cell_state, training=False):
        # TODO: document

        # Preparing
        out_concatenate = tf.concat((hidden_state, inputs), axis=1)

        forget_gate = self.dense_forget_gate(out_concatenate)

        input_gate = self.dense_input_gate(out_concatenate)

        state_candidates = self.dense_state_candidates(out_concatenate)

        # Update cell state
        cell_state = forget_gate * cell_state + input_gate * state_candidates

        # Determining the hidden state/output
        output_gate = self.dense_output(out_concatenate)

        hidden_state = output_gate * tf.math.tanh(cell_state)

        return hidden_state, cell_state


# Wrapper


class LSTM_Layer(tf.keras.layers.Layer):
    # TODO: document

    def __init__(self, cell_units):
        super(LSTM_Layer, self).__init__()

        self.cell = LSTM_Cell(cell_units)

    @tf.function
    def call(self, inputs, training=False):
        # TODO: document

        # https://www.tensorflow.org/guide/function#accumulating_values_in_a_loop
        length = inputs.shape[1]

        hidden_state = self.zero_states(inputs.shape[0])
        cell_state = self.zero_states(inputs.shape[0])

        hidden_states = tf.TensorArray(tf.float32, size=length)

        for t in tf.range(length):
            input_t = inputs[:, t, :]
            hidden_state, cell_state = self.cell(input_t, hidden_state, cell_state, training)
            hidden_states = hidden_states.write(t, hidden_state)        # cell_state here as well?

        return tf.transpose(hidden_states.stack(), [1, 0, 2])

    def zero_states(self, batch_size):
        # TODO: document
        return tf.zeros((batch_size, self.cell.state_size), tf.float32)


# Model


class LSTM_Model(tf.keras.Model):

    # TODO: document
    def __init__(self):
        super(LSTM_Model, self).__init__()

        self.lstm_layer = LSTM_Layer(1)
        self.out = tf.keras.layers.Dense(1, activation=tf.nn.sigmoid)

    def call(self, data, training=False):
        out_cell = self.lstm_layer(data, training)
        output = self.out(out_cell)

        return output
