import gym
from model import DQModel
import tensorflow as tf
from collections import deque
from data import prepare_data, prepare_state
from train_and_visualization import visualize, training, choose_action

# checkpoints saving path
checkpoint_path = "checkpoints/cp.ckpt"

# hyper-parameters
num_hidden_neurons = 200  # number of hidden layer neurons
batch_size = 8  # small batch to not overload the gpu
gamma = 0.99  # discount factor for reward
alpha = 0.000001  # scaling factor for thompson sampling
learning_rate = 0.01
max_memory = 100000
sample_size = 2000
num_new_frames = 500
epochs = 20000

# model initialization
dq_model = DQModel(num_hidden_neurons)
mse = tf.keras.losses.MeanSquaredError()
adam_optimizer = tf.keras.optimizers.Adam(learning_rate)

# experience replay buffer
memory = deque(maxlen=max_memory)

# tracker for losses and rewards
epoch_losses = []
epoch_rewards = []


# data generator
# picks random samples out of the erb
def gen():
    while True:
        # go from beginning to end and start new
        for i in range(len(memory)):
            yield memory[i]


# dataset generated out of generator data
# gets every episode new data
dataset = tf.data.Dataset.from_generator(
    gen,
    output_signature=(
        tf.TensorSpec(shape=(210, 160, 3), dtype=tf.float32),
        tf.TensorSpec(shape=(), dtype=tf.float32),
        tf.TensorSpec(shape=(), dtype=tf.float32),
        tf.TensorSpec(shape=(210, 160, 3), dtype=tf.float32),
    ),
)

# setup environment
env = gym.make(
    "ALE/Pong-v5",
    obs_type="rgb",  # ram | rgb | grayscale
    frameskip=5,  # frame skip
    mode=0,  # game mode, see Machado et al. 2018
    difficulty=0,  # game difficulty, see Machado et al. 2018
    repeat_action_probability=0.25,  # Sticky action probability
    full_action_space=True,  # Use all actions
    render_mode=None,  # None | human | rgb_array
)


frame_count = 0
state = env.reset()

# initializing experience replay buffer
while len(memory) < sample_size:

    # convert to tensor and normalize
    # don't overwrite state
    # otherwise there will be a different data format in the data generator
    prep_state = prepare_state(state)

    # choose action
    # for data preparation episode = 0
    action = choose_action(dq_model, prep_state, 0, epochs, alpha)

    # step the environment and add to experience replay buffer
    next_state, reward, done, _ = env.step(action)
    memory.append((state, action, reward, next_state))

    # next state is the state to make the next action from
    state = next_state
    frame_count += 1

    if done:  # about 1000 frames
        state = env.reset()  # reset env

    if reward != 0:  # Pong has either +1 or -1 reward exactly when game ends.
        print('frames %d: game finished, reward: %f' % (frame_count, reward) +
              ('' if reward == -1 else ' !!!!!!!!'))


# start training
for epoch in range(epochs):

    frame_count = 0
    reward_aggregator = []

    # prepare the samples
    memory_sample = prepare_data(dataset.take(sample_size), sample_size, batch_size)

    # train and track loss
    average_loss = training(memory_sample, dq_model, mse, adam_optimizer, gamma)

    # Save the weights using the `checkpoint_path` format
    dq_model.save_weights(checkpoint_path.format(epoch=epoch))

    epoch_losses.append(average_loss)

    print('ep %d: episode loss total was %f. running mean: %f'
          % (epoch, average_loss, tf.reduce_mean(epoch_losses)))

    # generating new samples
    while frame_count < num_new_frames:

        # convert to tensor and normalize
        # don't overwrite state
        # otherwise there will be a different data format in the data generator
        prep_state = prepare_state(state)

        # choose action
        action = choose_action(dq_model, prep_state, epoch, epochs, alpha)

        # step the environment and add to experience replay buffer
        next_state, reward, done, _ = env.step(action)
        memory.append((state, action, reward, next_state))

        # track reward
        # add only if a point was made
        if reward != 0:
            reward_aggregator.append(reward)

        # next state is the state to make the next action from
        state = next_state
        frame_count += 1

        if done:  # reset of environment
            state = env.reset()  # reset env

        if reward != 0:  # Pong has either +1 or -1 reward exactly when game ends.
            print('ep %d: game finished, reward: %f' % (epoch, reward) +
                  ('' if reward == -1 else ' !!!!!!!!'))

    # calculate average reward for the episode and print it
    # only if at least one game did end within the epoch
    if len(reward_aggregator) > 0:
        average_reward = tf.reduce_mean(reward_aggregator)
        epoch_rewards.append(average_reward)

        print('ep %d: episode reward total was %f. running mean: %f'
              % (epoch, average_reward, tf.reduce_mean(epoch_rewards)))


# plot average reward and loss per epoch
visualize(epoch_losses, epoch_rewards)
