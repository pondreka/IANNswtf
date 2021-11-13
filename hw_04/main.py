import pandas as pd
import tensorflow as tf


# ---------- Task 1
# -------- Task 1.1

wine_df = pd.read_csv("winequality-red.csv")

# >>> wine_df.columns
# Index(['fixed acidity;"volatile acidity";"citric acid";"residual sugar";"chlorides";"free sulfur dioxide";"total sulfur dioxide";"density";"pH";"sulphates";"alcohol";"quality"'], dtype='object')
