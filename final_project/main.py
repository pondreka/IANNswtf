import datetime
from ale_py import ALEInterface
from ale_py.roms import Pong
from gameplay import make_video, play_game
import gym
from model import DQN
import tensorflow as tf
import numpy as np



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
    # TODO: find where to place reset (apart form in the make_video)
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
