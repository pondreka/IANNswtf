import numpy as np


# ---------- task 2 "Data Set" ------------

inputs = np.array([[0, 0], [1, 0], [0, 1], [1, 1]])
outputs_AND = np.array([0, 0, 0, 1])
outputs_OR = np.array([0, 1, 1, 1])
outputs_NAND = np.array([1, 1, 1, 0])
outputs_NOR = np.array([1, 0, 0, 0])
outputs_XOR = np.array([0, 1, 1, 0])


def data_generator(labels: np.array) -> np.array:
    """Simple semi-random array generator function.

    Args:
        labels (np.array): the output labels to use for data.

    Returns:
        (np.array): a 1-D array with 3 entries. First two are inputs and
            the last one is the expected output (based on label).

    """
    while True:
        rand = np.random.randint(0, len(inputs))
        yield np.array([inputs[rand, 0], inputs[rand, 1], labels[rand]])
