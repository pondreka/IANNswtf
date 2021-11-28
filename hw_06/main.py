import tensorflow as tf
import matplotlib.pyplot as plt

# import numpy as np
import tensorflow_datasets as tfds

# from train_and_test import train_step, test
from custom_model import *
from data_preparation import prepare_data
from train_and_visualize import training, prepare_visualization

import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "True"


def main():
    # -------- task 1 "Data set" ---------
    ds_train_and_valid, ds_test = tfds.load(
        "cifar10",
        split=["train", "test"],
        shuffle_files=True,
        as_supervised=True,
    )

    # -------- task 1.1 "Construct a Data Pipeline" ------------

    overall_total: int = 50000
    valid_total: int = int(overall_total / 5)  # as many as test
    train_total: int = overall_total - valid_total
    # split the first batch
    ds_train = ds_train_and_valid.take(train_total)
    ds_valid = ds_train_and_valid.skip(train_total)

    # massage data
    train_ds = ds_train.apply(prepare_data)
    valid_ds = ds_valid.apply(prepare_data)
    test_ds = ds_test.apply(prepare_data)

    # -------- task 2 "Model" ------------
    test = False

    res_model = ResNet()
    dense_model = DenseNet()

    # -------- task 3 "Training" ------------

    num_epochs: int = 10
    learning_rate: float = 0.01

    # GOAL: achieve accuracy >= 85% (on test dataset)
    cat_cross_ent_loss = tf.keras.losses.CategoricalCrossentropy()
    # optimizer?
    adam_optimizer = tf.keras.optimizers.Adam(learning_rate)

    train, valid, test = training(
        model=res_model,
        loss=cat_cross_ent_loss,
        num_epochs=num_epochs,
        optimizer=adam_optimizer,
        train_ds=train_ds,
        valid_ds=valid_ds,
        test_ds=test_ds,
    )

    # ---------- task 4 "Visualization" ------------

    num_plot_visualization: int = 1
    _, axes = plt.subplots(
        nrows=num_plot_visualization * 2, ncols=1, sharex=True, figsize=(9, 6)
    )
    accuracies: int = 0
    losses: int = 1
    index: int = 0

    axes, index = prepare_visualization(
        axes,
        train[accuracies],
        train[losses],
        valid[accuracies],
        valid[losses],
        test[accuracies],
        test[losses],
        index,
        num_plot_visualization,
        group_name=f"epochs={num_epochs} " f"lr={learning_rate}, ",
        # f"batch={batch_size}, ",
        # f"dropout_rate={dropout_rate}",
    )

    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    main()
