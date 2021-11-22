import tensorflow as tf
import matplotlib.pyplot as plt
import numpy as np
import tensorflow_datasets as tfds
from train_and_test import train_step, test
from custom_model import ConvModel
from data_preparation import prepare_f_mnist_data


def main():
    # -------- task 1 "Data set" ---------
    ds_train_and_valid_fmnist, ds_test_fmnist = tfds.load(
        "fashion_mnist",
        split=["train", "test"],
        shuffle_files=True,
        as_supervised=True,
    )

    # -------- task 1.1 "Construct a Data Pipeline" ------------

    overall_total: int = 60000
    valid_total: int = int(overall_total / 6)  # as many as test
    train_total: int = overall_total - valid_total
    # split the first batch
    ds_train_fmnist = ds_train_and_valid_fmnist.take(train_total)
    ds_valid_fmnist = ds_train_and_valid_fmnist.skip(train_total)
    # massage data
    train_ds = ds_train_fmnist.apply(prepare_f_mnist_data)
    valid_ds = ds_valid_fmnist.apply(prepare_f_mnist_data)
    test_ds = ds_test_fmnist.apply(prepare_f_mnist_data)

    # -------- task 2 "Model" ------------

    model = ConvModel(shape=(28, 28))
    # TODO: add optimization stuff to model (like DropoutModel)?
    # dropout_rate = 0.1

    # -------- task 3 "Training" ------------

    num_epochs: int = 10
    learning_rate: float = 0.1
    
    # GOAL: achieve accuracy >= 85% (on test dataset)
    cat_cross_ent_loss = tf.keras.losses.CategoricalCrossentropy()
    # opmitizer?
    sgd_optimizer = tf.keras.optimizaters.SGD(learning_rate)

    train_pre, valid_pre, test_pre = training(
        model=model,
        loss=cat_cross_ent_loss,
        num_epochs=num_epochs,
        optimizer=sgd_optimizer,
        train_ds=train_ds,
        valid_ds=valid_ds,
        test_ds=test_ds,
    )


    # ---------- task 4 "Visualization" ------------
    # _, ax = plt.subplot(2, 1, sharex=True, figsize=(9, 6))
    #
    # ax[0].plot(train_losses, label="train loss")
    # ax[0].plot(test_losses, label="test loss")
    # ax[0].set(ylabel="Loss", title="Loss")
    # ax[0].legend(loc="upper right")
    #
    # ax[1].plot(train_accuracies, label="train accuracy")
    # ax[1].plot(test_accuracies, label="test accuracy")
    # ax[1].set(ylabel="Accuracy", title="Accuracy")
    # ax[1].legend(loc="upper right")
    #
    # plt.xlabel("Epochs")
    # plt.tight_layout()
    # plt.show()



if __name__ == "__main__":
    main()
