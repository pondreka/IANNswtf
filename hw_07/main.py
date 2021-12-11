import tensorflow as tf
import matplotlib.pyplot as plt

# import numpy as np
# import tensorflow_datasets as tfds

# from train_and_test import train_step, test
from custom_model import *
from data_preparation import prepare_data, my_integration_task
from train_and_visualize import training, prepare_visualization

import os

os.environ["KMP_DUPLICATE_LIB_OK"] = "True"


def main():
    # -------- task 1 "Data set" ---------

    data_set = tf.data.Dataset.from_generator(my_integration_task)

    # -------- task 1.3 "Construct a Data Pipeline" ------------

    overall_total: int = 50000
    valid_total: int = int(overall_total / 5)  # as many as test
    train_total: int = overall_total - (valid_total * 2)
    # split the first batch
    ds_train = ds_train_and_valid.take(train_total)
    ds_valid = ds_train_and_valid.skip(train_total).take(valid_total)
    ds_test = ds_train_and_valid.skip(train_total + valid_total).take(
        valid_total
    )

    # massage data
    train_ds = ds_train.apply(prepare_data)
    valid_ds = ds_valid.apply(prepare_data)
    test_ds = ds_test.apply(prepare_data)

    # -------- task 2 "Model" ------------

    res_model = ResNet()
    dense_model = DenseNet()

    # -------- task 3 "Training" ------------

    num_epochs: int = 3
    learning_rate: float = 0.001

    cat_cross_ent_loss = tf.keras.losses.CategoricalCrossentropy()
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

    train2, valid2, test2 = training(
        model=dense_model,
        loss=cat_cross_ent_loss,
        num_epochs=num_epochs,
        optimizer=adam_optimizer,
        train_ds=train_ds,
        valid_ds=valid_ds,
        test_ds=test_ds,
    )

    # ---------- task 4 "Visualization" ------------
    res_model.summary()
    dense_model.summary()
    num_plot_visualization: int = 2
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
        group_name=f"epochs={num_epochs} lr={learning_rate} | accur.={test[accuracies][-1]}",
        # f"batch={batch_size}, ",
        # f"dropout_rate={dropout_rate}",
    )

    axes, index = prepare_visualization(
        axes,
        train2[accuracies],
        train2[losses],
        valid2[accuracies],
        valid2[losses],
        test2[accuracies],
        test2[losses],
        index,
        num_plot_visualization,
        group_name=f"epochs={num_epochs} lr={learning_rate} | accur.={test2[accuracies][-1]}",
        # f"batch={batch_size}, ",
        # f"dropout_rate={dropout_rate}",
    )

    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    main()
