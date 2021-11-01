import numpy as np
from Sigmoid import sigmoid


# ------------- task 3 "Perceptron" -------------
class Perceptron:
    """Instance representation of a single perceptron."""
    def __init__(self, input_units: int):
        # create weights for incoming input_units
        self.weights = np.random.randn(input_units)
        # add bias for this perceptron
        self.bias = np.random.randn()
        # learning rate
        self.alpha = 1
        # the actual incoming inputs (will be updated every forward ste)
        self.inputs = np.zeros(input_units)

    def forward_step(self, inputs: np.array) -> float:
        """Calculate perceptron activation.

        Args:
            inputs (np.array): the incoming array of inputs trying
                to activate this perceptron.

        Returns:
            (float): the final activation of the perceptron.

        """

        # calculate net input
        drive = self.bias
        for i in range(len(inputs)):
            drive += inputs[i] * self.weights[i]

        # Update our saved inputs
        self.inputs = inputs

        return sigmoid(drive)

    def update(self, delta):
        """Helper function to update bias and weights."""

        # update bias
        self.bias -= self.alpha * delta

        # update weights
        for i in range(len(self.weights)):
            self.weights[i] -= self.alpha * delta * self.inputs[i]
