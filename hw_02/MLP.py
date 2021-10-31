import numpy as np
from Sigmoid import sigmoidprime
from Perceptron import Perceptron


# task 4 "Multi-Layer Perceptron"
class MLP:
    """Multi-Layer Perceptron instance.

    Specific for classifying inputs for logic gates.
    """
    def __init__(self):
        self.hiddenLayer = []
        # only 4 neurons needed in the hidden layer
        self._ps_in_hl: int = 4
        for _ in range(self._ps_in_hl):
            # two inputs per neuron
            perceptron = Perceptron(2)
            self.hiddenLayer.append(perceptron)
        # one output neuron, with 4 inputs comming from hidden layer.
        self.outputPerceptron = Perceptron(self._ps_in_hl)
        self.output = 0
        # 2-D array with inputs from before and after forward step
        self.ins = []

    def forward_step(self, inputs: np.array) -> None:
        """Pass inputs into the network."""
        self.ins.append(inputs)

        new_inputs = np.zeros(self._ps_in_hl)
        for i in range(self._ps_in_hl):
            new_inputs[i] = self.hiddenLayer[i].forward_step(inputs)

        self.ins.append(new_inputs)

        self.output = self.outputPerceptron.forward_step(new_inputs)

    # IMPORTANT CHANGES: don't pass `self.ins[0]` to `update` since
    # inputs are already saved per perceptron.
    def backprop_step(self, target: float) -> None:
        """Update parameters (delta and weights) on network."""
        deltaN = -(target - self.output) * self.output * (1 - self.output)

        self.outputPerceptron.update(deltaN)

        e = self.outputPerceptron.bias * deltaN
        for i in range(len(self.outputPerceptron.weights)):
            e += deltaN * self.outputPerceptron.weights[i]

        for i in range(len(self.hiddenLayer)):
            delta = e * self.ins[1][i] * (1 - self.ins[1][i])
            # IMPORTANT CHANGES just apply to this one line here
            self.hiddenLayer[i].update(delta)

        self.ins = []
