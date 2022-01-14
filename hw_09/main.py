from data import load_data
from model import Discriminator, Generator
from train_and_test import train_step, test_step
import tensorflow as tf


# Press the green button in the gutter to run the script.
if __name__ == '__main__':

    # ------------ task 1 "Data set" -----------------
    train, test = load_data()

    # ------------- task 2 "Model" -------------------
    discriminator = Discriminator()
    generator = Generator()

    # ------------- task 3 "Training" ----------------
    num_epochs = 10

    for epoch in range(1, num_epochs + 1):
        print(f"Epoch {epoch}:")

        for data in train:
            metrics = train_step(discriminator, generator, data)

        print([f"{key}: {value}" for (key, value) in zip(list(metrics.keys()), list(metrics.values()))])

        discriminator.reset_metrics()
        generator.reset_metrics()

        for data in test:
            metrics = test_step(discriminator, generator, data)

        print([f"test_{key}: {value}" for (key, value) in zip(list(metrics.keys()), list(metrics.values()))])

        discriminator.reset_metrics()
        generator.reset_metrics()

