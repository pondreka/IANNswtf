import numpy as np


# ---------- task 1 "Preparation" -----------
def sigmoid(x: float) -> float:
    """Basic sigmoid function."""
    return 1 / (1 + np.exp(-x))


def sigmoidprime(x: float) -> float:
    """Derivative of the sigmoid function."""
    return sigmoid(x) * (1 - sigmoid(x))
