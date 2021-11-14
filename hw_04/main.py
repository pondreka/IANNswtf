import pandas as pd
import tensorflow as tf


# ---------- Task 1
# -------- Task 1.1

wine_df = pd.read_csv("winequality-red.csv", sep=";")

# >>> wine_df.columns
# Index(['fixed acidity;"volatile acidity";"citric acid";"residual sugar";"chlorides";"free sulfur dioxide";"total sulfur dioxide";"density";"pH";"sulphates";"alcohol";"quality"'], dtype='object')
# make df into ds
full_size: int = len(wine_df)
train_size = int(0.7 * full_size)
valid_size = int(0.15 * full_size)

labels = tf.squeeze([wine_df.pop("quality")], axis=0)
full_ds = tf.data.Dataset.from_tensor_slices((dict(wine_df)), labels)
train_ds = full_ds.take(train_size)
remaining = full_ds.skip(train_size)
valid_ds = remaining.take(valid_size)
test_ds = remaining.skip(valid_size)
