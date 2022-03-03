import gym
from ale_py import ALEInterface
from ale_py.roms import Pong
from model import DQN
from statistics import mean
import os
import datetime


def play_game(env, train_network, target_network, epsilon, copy_step):
    """Function to make network play game."""
    rewards: float = 0
    iterations: int = 0
    done: bool = False
    observations = env.reset()
    losses: list = []
    while not done:
        action = train_network.get_action(observations, epsilon)
        prev_observations = observations
        observations, reward, done, _ = env.step(action)
        rewards += reward
        if done:
            reward = -200
            env.reset()

        exp: dict = {
            "s": prev_observations,
            "a": action,
            "r": reward,
            "s2": observations,
            "done": done,
        }
        train_network.add_experience(exp)
        loss = train_network.train(target_network)
        if isinstance(loss, int):
            losses.append(loss)
        else:
            losses.append(loss.numpy())
        iterations += 1
        if iterations % copy_step == 0:
            target_network.copy_weights(train_network)
    return rewards, mean(losses)


def make_video(env, train_network):
    """Dummy doc"""
    env = gym.wrappers.Monitor(
        env, os.path.join(os.getcwd(), "videos"), force=True
    )
    rewards: float = 0
    steps: int = 0
    done: bool = False
    observation = env.reset()
    while not done:
        action = train_network.get_action(observation, 0)
        observation, reward, done, _ = env.step(action)
        steps += 1
        rewards += reward
    print(f"Testing steps: {steps} rewards {rewards}: ")


def main():
    """Main function."""
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
    # TODO: find where to place reset
    # env.reset()
    #     env.step(env.action_space.sample())  # take a random action
    ale = ALEInterface()
    ale.loadROM(Pong)
    gamma: float = 0.99
    copy_step: int = 25
    num_states: int = len(env.observation_space.sample())
    num_actions: int = env.action_space.n
    hidden_units: tuple = (200, 200)
    max_experiences: int = 10000
    min_experiences: int = 100
    batch_size: int = 32
    learn_rate: float = 1e-2
    current_time = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
    log_dir: str = f"logs/dqn/{current_time}"
    summary_writer = tf.summary.create_file_writer(log_dir)

    train_network = DQN(
        num_states,
        num_actions,
        hidden_units,
        gamma,
        max_experiences,
        min_experiences,
        batch_size,
        learn_rate,
    )
    target_network = DQN(
        num_states,
        num_actions,
        hidden_units,
        gamma,
        max_experiences,
        min_experiences,
        batch_size,
        learn_rate,
    )
    big_n: int = 50000
    total_rewards = np.empty(big_n)
    epsilon: float = 0.99
    decay: float = 0.9999
    min_epsilon: float = 0.1
    for _t in range(big_n):
        epsilon = max(min_epsilon, epsilon * decay)
        total_reward, losses = play_game(
            env, train_network, target_network, epsilon, copy_step
        )
        total_rewards[_t] = total_reward
        avg_rewards = total_rewards[max(0, _t - 100) : (_t + 1)].mean()
        with summary_writer.as_default():
            tf.summary.scalar("episode reward", total_reward, step=_t)
            tf.summary.scalar("running avg reward(100)", avg_rewards, step=_t)
            tf.summary.scalar("average loss)", losses, step=_t)
        if _t % 100 == 0:
            print(
                "episode:",
                _t,
                "episode reward:",
                total_reward,
                "eps:",
                epsilon,
                "avg reward (last 100):",
                avg_rewards,
                "episode loss: ",
                losses,
            )
    print("avg reward for last 100 episodes:", avg_rewards)
    make_video(env, train_network)
    env.close()


if __name__ == "__main__":
    main()
