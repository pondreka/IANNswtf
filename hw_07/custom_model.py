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

        # self.bias = tf.Variable(tf.zeros(units), name="LSTM_Cell_biases")

        self.state_size = units

    @tf.function
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
    # TODO: document

    def __init__(self, cell_units):
        super(LSTM_Layer, self).__init__()

        self.cell_units = cell_units
        self.cell = LSTM_Cell(cell_units)

    @tf.function
    def call(self, x, states, training=False):
        # TODO: document

        # https://www.tensorflow.org/guide/function#accumulating_values_in_a_loop
        length = x.shape[1]

        states = tf.TensorArray(tf.float32, size=length)

        for i in tf.range(length):
            state = self.cell(input_data[i], states)
            states = states.write(i, state[0])

        return tf.transpose(states.stack(), [1, 0, 2])

    def zero_states(self, batch_size):
        # TODO: document
        return (
            tf.zeros(self.cell_units, batch_size),
            tf.zeros(self.cell_units, batch_size),
        )


# Model


class LSTM_Model(tf.keras.Model):
    def __init__(self, units):
        super(LSTM_Model, self).__init__()

        self.lstm_layer = LSTM_Layer(units)

    def call(self, data, training=False):
        x = self.lstm_layer(data)
        x = self.dense(x)
        x = self.out(x)

        return x
