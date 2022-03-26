import gym
import random
from ale_py import ALEInterface
from ale_py.roms import Pong
from model import DQModel
import tensorflow as tf
from collections import deque
from data import prepare_state, calculate_targets

# hyperparameters
H = 200  # number of hidden layer neurons
batch_size = 8  # samll batch to not overload the gpu
gamma = 0.99  # discount factor for reward
learning_rate: float = 0.01
max_memory = 100000  # change to 100000
sample_size = 10000  # change to 10000
alpha = 0.000001  # scaling factor for thompson sampling
episodes = 10000  # change to 10000

# model initialization
dq_model = DQModel(H)
mse = tf.keras.losses.MeanSquaredError()
adam_optimizer = tf.keras.optimizers.Adam(learning_rate)

# experience replay buffer
memory = deque(maxlen=max_memory)


# data generator
# picks random samples out of the erb
def gen():
    while True:
        # go from beginning to end and start new
        for i in range(len(memory)):
            yield memory[i]


# def gen():
#     while True:
#         # random selection
#         rand = random.randint(0, max_memory - 1)
#         yield memory[rand]


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


# choose action by thompson sampling
def choose_action(model, x, episode):

    if episode < 5000:
        temperature = 1  # high temperature (low value) produces nearly probabilities of 0.5
    else:
        episode = tf.cast(episode, tf.float32)
        temperature = 1 + alpha * tf.math.log(episode)  # getting lower temperature (higher values) for later episodes
                                                        # for more differenciated probabilities

    p = model(x)
    probs = tf.nn.softmax(p * temperature)  # thompson sampling
    action = int(tf.random.categorical(probs, 1))  # get action dependent on higher probability

    return action + 3  # add 3 to get correct action index (3 = up, 4 = down)


# train based on data samples from erb
def training(data, model, loss_function, optimizer, gamma):

    for s, a, r, n in data:
        # calculate q values for next state
        q_values = model(n)
        # calculate target by target = reward + gamma * argmax(q_values)
        targets = calculate_targets(q_values, r, gamma)

        with tf.GradientTape() as tape:
            # make predictions for current state
            q_preds = model(s)
            # calculate loss between taget and q prediction for the action
            loss = loss_function(targets, tf.reduce_sum(q_preds * a, axis=-1))
            # update gradient and optimizer
            gradients = tape.gradient(loss, model.trainable_variables)
        optimizer.apply_gradients(zip(gradients, model.trainable_variables))


# setup environment
env = gym.make(
    "ALE/Pong-v5",
    obs_type="rgb",  # ram | rgb | grayscale
    frameskip=5,  # frame skip
    mode=0,  # game mode, see Machado et al. 2018
    difficulty=0,  # game difficulty, see Machado et al. 2018
    repeat_action_probability=0.25,  # Sticky action probability
    full_action_space=True,  # Use all actions
    render_mode="human",  # None | human | rgb_array
)

# render environment
ale = ALEInterface()
ale.loadROM(Pong)

frame_count = 0
state = env.reset()
running_reward = None
reward_sum = 0

# initializing experience replay buffer
# fill with 20000 samples first
while len(memory) < sample_size * 2:

    # convert to tensor and normalize
    # don't overwrite state
    # otherwise there will be a different data format in the datagenerator
    prep_state = prepare_state(state)

    # choose action
    # for data preparation episode = 0
    action = choose_action(dq_model, prep_state, 0)

    # step the environment and add to experience replay buffer
    next_state, reward, done, _ = env.step(action)
    memory.append((state, action, reward, next_state))

    # next state is the state to make the next action from
    state = next_state

    reward_sum += reward
    frame_count += 1

    if done:  # about 1000 frames

        # boring book-keeping
        running_reward = reward_sum if running_reward is None else running_reward * 0.99 + reward_sum * 0.01
        print('resetting env. episode reward total was %f. running mean: %f' % (reward_sum, running_reward))

        reward_sum = 0
        state = env.reset()  # reset env

    if reward != 0:  # Pong has either +1 or -1 reward exactly when game ends.
        print('frames %d: game finished, reward: %f' % (frame_count, reward) + ('' if reward == -1 else ' !!!!!!!!'))

# todo: save memory

# actual training
frame_count = 0
state = env.reset()
running_reward = None
reward_sum = 0

for episode in range(episodes):

    frame_count = 0

    ### data pipeline
    # batching did somehow not work in a function
    # normalize image values
    # one hot encode actions
    memory_sample = dataset.take(sample_size)\
        .map(lambda s, a, r, n: (s / 255, a, r, n / 255))\
        .map(lambda s, a, r, n: (s, tf.where(tf.equal(a, 3), [1.0, 0.0], [0.0, 1.0]), r, n))\
        .cache().shuffle(1000).batch(batch_size, drop_remainder=True).prefetch(16)

    ### training
    training(memory_sample, dq_model, mse, adam_optimizer, gamma)
    # todo: save weights

    # every 500 episodes sample new
    if episode % 500 == 0:

        ### generating new samples
        while frame_count < sample_size:

            # convert to tensor and normalize
            # don't overwrite state
            # otherwise there will be a different data format in the datagenerator
            prep_state = prepare_state(state)

            # choose action
            action = choose_action(dq_model, prep_state, episode)

            # step the environment and add to experience replay buffer
            next_state, reward, done, _ = env.step(action)
            memory.append((state, action, reward, next_state))

            # next state is the state to make the next action from
            state = next_state

            reward_sum += reward
            frame_count += 1

            if done:  # episode ends
                # todo: save samples and convert to video or gif
                # boring book-keeping
                running_reward = reward_sum if running_reward is None else running_reward * 0.99 + reward_sum * 0.01
                print('resetting env. episode reward total was %f. running mean: %f' % (reward_sum, running_reward))

                reward_sum = 0
                state = env.reset()  # reset env

            if reward != 0:  # Pong has either +1 or -1 reward exactly when game ends.
                print(
                    'ep %d: game finished, reward: %f' % (episode, reward) + ('' if reward == -1 else ' !!!!!!!!'))

