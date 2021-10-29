import numpy as np
from Sigmoid import sigmoid


# task 3 "Perceptron"
class Perceptron:

    def __init__(self, input_units):
        self.weights = np.random.randn(input_units)
        self.bias = np.random.randn()
        self.alpha = 1
        self.inputs = np.zeros(input_units)

    def forward_step(self, inputs):

        drive = self.bias
        for i in range(len(inputs)):
            drive += inputs[i] * self.weights[i]

        return sigmoid(drive)

    def update(self, delta, inputs):

        self.bias -= self.alpha * delta

        for i in range(len(self.weights)):
            self.weights[i] -= self.alpha * delta * inputs[i]


