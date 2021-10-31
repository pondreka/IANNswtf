import numpy as np
from Sigmoid import sigmoidprime
from Perceptron import Perceptron


# task 4 "Multi-Layer Perceptron"
class MLP:
    def __init__(self):

        self.hiddenLayer = []
        for i in range(4):
            perceptron = Perceptron(2)
            self.hiddenLayer.append(perceptron)

        self.outputPerceptron = Perceptron(4)

        self.output = 0
        self.ins = []

    def forward_step(self, inputs):

        self.ins.append(inputs)

        new_inputs = np.zeros(4)
        for i in range(len(self.hiddenLayer)):
            new_inputs[i] = self.hiddenLayer[i].forward_step(inputs)

        self.ins.append(new_inputs)

        self.output = self.outputPerceptron.forward_step(new_inputs)

    def backprop_step(self, target):

        deltaN = -(target - self.output) * self.output * (1 - self.output)

        self.outputPerceptron.update(deltaN, self.ins[1])

        e = self.outputPerceptron.bias * deltaN
        for i in range(len(self.outputPerceptron.weights)):
            e += deltaN * self.outputPerceptron.weights[i]

        for i in range(len(self.hiddenLayer)):
            delta = e * self.ins[1][i] * (1 - self.ins[1][i])
            self.hiddenLayer[i].update(delta, self.ins[0])

        self.ins = []
