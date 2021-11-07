import numpy as np
from matplotlib import pyplot as plt
from Dataset import data_generator, outputs_AND, outputs_OR, outputs_NAND, outputs_NOR, outputs_XOR
from MLP import MLP

DATASET_SIZE = 100
TRAINING_EPOCH = 1000


def main():
    # datasets generation depending on the labeling
    # Note: input can be changed for any of the imported outputs from Dataset
    data_gen = data_generator(outputs_OR)

    # Creating an artificial dataset for the chosen label function on top
    data_set = [next(data_gen) for _ in range(DATASET_SIZE)]

    # --------- task 5 "Training" ------------
    mlp = MLP()
    av_loss = []
    av_accuracy = []

    for _ in range(TRAINING_EPOCH):

        loss = []
        accuracy = []
        correct = 0

        for count, p in enumerate(data_set):

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
    x = np.arange(TRAINING_EPOCH)
    fig, axs = plt.subplots(2)
    fig.suptitle("MLP Accuracy and Loss")

    # accuracy plotting
    axs[0].plot(x, av_accuracy, "ro")
    axs[0].set(ylabel="accuracy")

    # loss plotting
    axs[1].plot(x, av_loss, "ro")
    axs[1].set(ylabel="loss")
    plt.show()


if __name__ == "__main__":
    main()
