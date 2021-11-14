import pandas as pd
import tensorflow as tf
import numpy as np
import custom_model
from tests import test, train_step
import matplotlib.pyplot as plt

# ---------- Task 1
# -------- Task 1.1

def prepare_ds(dataset, take_amount: int = 100000):
    """Prepares dataset with pipelines and returns specified amount"""
    # dataset = dataset.take(take_amount)
    dataset = dataset.map(
        lambda seq, label: (one_hot_converter(seq), tf.one_hot(label, 10))
    )
    dataset = dataset.cache()
    dataset = dataset.shuffle(1000)
    dataset = dataset.batch(128)
    dataset = dataset.prefetch(10)
    return dataset


def df_to_dataset(df, batch_size=32):
    df = df.copy()
    labels = tf.squeeze(tf.constant([df.pop("quality")]), axis=0)
    ds = tf.data.Dataset.from_tensor_slices(
        (df, labels)
    )
    ds = ds.map(
        lambda input, labels: (input, labels > 5)
    )
    ds = ds.cache()
    ds = ds.shuffle(128)
    ds = ds.batch(batch_size)
    ds = ds.prefetch(16)
    return ds

wine_df = pd.read_csv("winequality-red.csv", sep=";")

# >>> wine_df.columns
# Index(['fixed acidity;"volatile acidity";"citric acid";"residual sugar";"chlorides";"free sulfur dioxide";"total sulfur dioxide";"density";"pH";"sulphates";"alcohol";"quality"'], dtype='object')
# make df into ds
full_size: int = len(wine_df)
train_size = int(0.7 * full_size)
valid_size = int(0.15 * full_size)

ran_stat = np.random.RandomState()
train_df = wine_df.sample(frac=0.7, random_state=ran_stat)
wine_df = wine_df.drop(train_df.index, axis=0)
valid_df = wine_df.sample(frac=0.5, random_state=ran_stat)
test_df = wine_df.drop(valid_df.index, axis=0)


train_ds = df_to_dataset(train_df)
valid_ds = df_to_dataset(valid_df)
test_ds = df_to_dataset(test_df)


# Task 2: Model

super_model = custom_model.CustomModel()

# Task 3: Training

num_epochs = 100
learning_rate = 0.0001

binary_cross_entropy = tf.keras.losses.BinaryCrossentropy()
# Super optimizer
optimizer = tf.keras.optimizers.Adam(learning_rate)

# Prepare some data for the final visualization
# lists of tensors
train_losses: list = []
train_accuracies: list = []

test_losses: list = []
test_accuracies: list = []

valid_losses: list = []
valid_accuracies: list = []


# testing once before we begin
test_loss, test_accuracy = test(super_model, test_ds, binary_cross_entropy)
test_losses.append(test_loss)
test_accuracies.append(test_accuracy)

# check how model performs on train data once before we begin
train_loss, train_accuracy = test(super_model, train_ds, binary_cross_entropy)
train_losses.append(train_loss)
train_accuracies.append(train_accuracy)

# testing once before we begin
valid_loss, valid_accuracy = test(super_model, valid_ds, binary_cross_entropy)
valid_losses.append(valid_loss)
valid_accuracies.append(valid_accuracy)
# We train for num_epochs epochs.
for epoch in range(num_epochs):
    print(
        f"Epoch {str(epoch)}: \tTrain accuracy:{train_accuracies[-1]}\n"
        f"\t\tTest accuracy {test_accuracies[-1]}"
    )

    # training (and checking in with training)
    epoch_loss_agg = []
    epoch_accuracy_agg = []
    for input, target in train_ds:
        train_loss, train_accuracy = train_step(
            super_model, input, target, binary_cross_entropy, optimizer
        )
        epoch_loss_agg.append(train_loss)
        epoch_accuracy_agg.append(train_accuracy)

    # track training loss
    train_losses.append(tf.reduce_mean(epoch_loss_agg))
    train_accuracies.append(tf.reduce_mean(epoch_accuracy_agg))

    # testing, so we can track accuracy and test loss
    test_loss, test_accuracy = test(super_model, test_ds, binary_cross_entropy)
    test_losses.append(test_loss)
    test_accuracies.append(test_accuracy)

# ------------ task 4 "Visualization" ----------------

# Visualize accuracy and loss for training and test data.
plt.figure()
(line0,) = plt.plot(train_losses)
(line1,) = plt.plot(train_accuracies)
(line2,) = plt.plot(test_losses)
(line3,) = plt.plot(test_accuracies)
plt.xlabel("Training steps")
plt.ylabel("Loss/Accuracy")
plt.legend(
    (line0, line1, line2, line3),
    ("training", "training accuracy", "test", "test accuracy"),
)
plt.show()
