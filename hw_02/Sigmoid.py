import numpy as np


# task 1 "Preparation"
def sigmoid(x):
    return 1 / (1 + np.exp(-x))


def sigmoidprime(x):
    return sigmoid(x) * (1 - sigmoid(x))





