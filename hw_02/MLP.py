import numpy as np
from Perceptron import Perceptron


# ----------- task 4 "Multi-Layer Perceptron" ----------
class MLP:
    """
    Multi-Layer Perceptron instance.

    Specific for classifying inputs for logic gates.
    """
    def __init__(self):
        self.hidden_layer = []
        # only 4 neurons needed in the hidden layer
        self._ps_in_hl: int = 4
        for _ in range(self._ps_in_hl):
            # two inputs per neuron
            perceptron = Perceptron(2)
            self.hidden_layer.append(perceptron)
        # one output neuron, with 4 inputs coming from hidden layer.
        self.output_perceptron = Perceptron(self._ps_in_hl)
        # output value
        self.output = 0

    def forward_step(self, inputs: np.array) -> None:
        """Pass inputs into the network."""

        new_inputs = np.zeros(self._ps_in_hl)

        # activation of the hidden layer
        for i in range(self._ps_in_hl):
            new_inputs[i] = self.hidden_layer[i].forward_step(inputs)

        # activation of the output layer
        self.output = self.output_perceptron.forward_step(new_inputs)

    def backprop_step(self, target: float) -> None:
        """Update parameters (delta and weights) on network."""

        # update output layer
        deltaN = -(target - self.output) * self.output * (1 - self.output)
        self.output_perceptron.update(deltaN)

        # update error for the new layer
        e = self.output_perceptron.bias * deltaN
        for i in range(len(self.output_perceptron.weights)):
            e += deltaN * self.output_perceptron.weights[i]

        # update hidden layer perceptrons
        for i in range(len(self.hidden_layer)):
            delta = e * self.output_perceptron.inputs[i] * (1 - self.output_perceptron.inputs[i])
            self.hidden_layer[i].update(delta)

