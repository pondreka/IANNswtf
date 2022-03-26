import tensorflow as tf


# get the state for picking the next action in the right shape to fit through the model
def prepare_state(state):

    state = tf.convert_to_tensor(state)
    state = tf.cast(state, tf.float32)
    state = state / 128 - 1
    state = tf.expand_dims(state, axis=0)

    return state


# calculate q target
def calculate_targets(q_values, rewards, gamma):
    q_targets = rewards + gamma * tf.reduce_max(q_values)
    return q_targets
