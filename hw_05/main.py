import tensorflow as tf
import matplotlib.pyplot as plt
import tensorflow_datasets as tfds

from custom_model import ConvModel
from data_preparation import prepare_f_mnist_data
from train_and_visualize import training, prepare_visualization


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

    model = ConvModel()

    # -------- task 3 "Training" ------------

    num_epochs: int = 10
    learning_rate: float = 0.01

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
        group_name=f"epochs={num_epochs} " 
                   f"lr={learning_rate}, "
    )

    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    main()
