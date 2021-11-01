import numpy as np
from MLP import MLP
import matplotlib.pyplot as plt


# ---------- task 2 "Data Set" ------------

inputs = np.array([[0, 0], [1, 0], [0, 1], [1, 1]])
outputs_AND = np.array([0, 0, 0, 1])
outputs_OR = np.array([0, 1, 1, 1])
outputs_NAND = np.array([0, 1, 1, 1])
outputs_NOR = np.array([1, 0, 0, 0])
outputs_XOR = np.array([0, 1, 1, 0])


def dataGenerator(labels: np.array) -> np.array:
    """Simple semi-random array generator function.

    Args:
        labels (np.array): the output labels to use for data.

    Returns:
        (np.array): a 1-D array with 3 entries. First two are inputs and
            the last one is the expected output (based on label).

    """
    while True:
        rand = np.random.randint(0, 4)
        yield np.array([inputs[rand, 0], inputs[rand, 1], labels[rand]])


# different datasets depending on the labeling
dataGen = dataGenerator(outputs_AND)
# dataGen = dataGenerator(outputs_OR)
# dataGen = dataGenerator(outputs_NAND)
# dataGen = dataGenerator(outputs_NOR)
# dataGen = dataGenerator(outputs_XOR)

dataSet = []

# Creating an artificial dataset for the chosen label function on top
for _ in range(100):
    dataSet.append(next(dataGen))


# --------- task 5 "Training" ------------
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

        # forward step
        mlp.forward_step(inputs)

        # loss calculation
        loss.append((target - mlp.output) ** 2)

        # counting for correct labeling
        if (target == 1 and mlp.output > 0.5) ^ (
            target == 0 and mlp.output <= 0.5
        ):
            correct += 1

        # accuracy calculation
        accuracy.append(correct / (count + 1))

        # backpropagation step
        mlp.backprop_step(target)

    # average loss calculation
    loss_a = np.array(loss)
    mean_loss = np.mean(loss_a)
    av_loss.append(mean_loss)

    # average accuracy calculation
    accuracy_a = np.array(accuracy)
    mean_accuracy = np.mean(accuracy_a)
    av_accuracy.append(mean_accuracy)


# -------- task 6 "Visualization" ------------
x = np.arange(1000)
fig, axs = plt.subplots(2)
fig.suptitle("MLP Accuracy and Loss")

# accuracy plotting
axs[0].plot(x, av_accuracy, "ro")
axs[0].set(ylabel="accuracy")

# loss plotting
axs[1].plot(x, av_loss, "ro")
axs[1].set(ylabel="loss")
plt.show()
