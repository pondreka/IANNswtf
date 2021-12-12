import tensorflow as tf
import matplotlib.pyplot as plt

# import numpy as np
# import tensorflow_datasets as tfds

# from train_and_test import train_step, test
from custom_model import LSTM_Model
from data_preparation import prepare_data, create_dataset
from train_and_visualize import training, prepare_visualization

import os

os.environ["KMP_DUPLICATE_LIB_OK"] = "True"


def main():
    # -------- task 1.2 "Generate a Tensorflow Dataset" ---------

    data_set = create_dataset()

    # -------- task 1.3 "Create a Data pipeline" ------------

    ds_train = data_set.take(7000)
    ds_valid = data_set.take(1500)
    ds_test = data_set.take(1500)

    # massage data
    train_ds = ds_train.apply(prepare_data)
    valid_ds = ds_valid.apply(prepare_data)
    test_ds = ds_test.apply(prepare_data)

    # -------- task 2 "The network" ------------

    model = LSTM_Model()

    # -------- task 3 "Training" ------------

    num_epochs: int = 15
    learning_rate: float = 0.001

    cat_cross_ent_loss = tf.keras.losses.CategoricalCrossentropy()
    adam_optimizer = tf.keras.optimizers.Adam(learning_rate)

    train, valid, test = training(
        model=model,
        loss=cat_cross_ent_loss,
        num_epochs=num_epochs,
        optimizer=adam_optimizer,
        train_ds=train_ds,
        valid_ds=valid_ds,
        test_ds=test_ds,
    )

    # ---------- task 4 "Visualization" ------------
    model.summary()

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
        group_name=f"epochs={num_epochs} lr={learning_rate} | accur.={test[accuracies][-1]}",
        # f"batch={batch_size}, ",
        # f"dropout_rate={dropout_rate}",
    )

    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    main()
