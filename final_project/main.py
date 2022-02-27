import gym
from ale_py import ALEInterface
from ale_py.roms import Pong
from model import DQN


def main():
    """Function to call at beginning of program."""
    ale = ALEInterface()
    ale.loadROM(Pong)

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
    env.reset()
    for _ in range(1000):
        env.step(env.action_space.sample())  # take a random action
    env.close()


if __name__ == "__main__":
    main()
