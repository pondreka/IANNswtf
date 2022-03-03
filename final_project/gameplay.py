from statistics import mean
import os
import gym


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
    """Provide a video of the network playing the game.

    After training the network, it's good to test how it performed
    by checking gameplay form it.
    """
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
