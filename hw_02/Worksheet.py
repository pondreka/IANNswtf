import numpy as np
from MLP import MLP
import matplotlib.pyplot as plt


# task 2 "Data Set"

inputs = np.array([[0, 0], [1, 0], [0, 1], [1, 1]])
outputs_AND = np.array([0, 0, 0, 1])
outputs_OR = np.array([0, 1, 1, 1])
outputs_NAND = np.array([0, 1, 1, 1])
outputs_NOR = np.array([1, 0, 0, 0])
outputs_XOR = np.array([0, 1, 1, 0])


def dataGenerator(labels):
    while True:
        rand = np.random.randint(0, 4)
        yield np.array([inputs[rand, 0], inputs[rand, 1], labels[rand]])


dataGen = dataGenerator(outputs_AND)
# dataGen = dataGenerator(outputs_OR)
# dataGen = dataGenerator(outputs_NAND)
# dataGen = dataGenerator(outputs_NOR)
# dataGen = dataGenerator(outputs_XOR)

dataSet = []

for _ in range(10):
    dataSet.append(next(dataGen))


# task 5 "Training"
mlp = MLP()
av_loss = []
av_accuracy = []

for _ in range(1000):

    loss = []
    accuracy = []
    correct = 0

    for count, p in enumerate(dataSet):

        inputs = np.array([p[0], p[1]])
        target = p[2]

        mlp.forward_step(inputs)

        l = (target - mlp.output) ** 2
        loss.append(l)

        if (target == 1 and mlp.output > 0.5) ^ (
            target == 0 and mlp.output <= 0.5
        ):
            correct += 1

        a = correct / (count + 1)
        accuracy.append(a)

        mlp.backprop_step(target)

    loss_a = np.array(loss)
    mean_loss = np.mean(loss_a)
    av_loss.append(mean_loss)

    accuracy_a = np.array(accuracy)
    mean_accuracy = np.mean(accuracy_a)
    av_accuracy.append(mean_accuracy)


# task 6 "Visualization"
x = np.arange(1000)
plt.plot(x, av_accuracy, "ro")
plt.ylabel("accuracy")
plt.axis([0, 1000, 0, 1])
plt.show()

plt.plot(x, av_loss, "ro")
plt.ylabel("loss")
plt.axis([0, 1000, 0, 1])
plt.show()
