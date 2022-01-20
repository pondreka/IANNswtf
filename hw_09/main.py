from data import load_data
from model import Discriminator, Generator
from train_and_test import train_step, test_step
import tensorflow as tf
import matplotlib.pyplot as plt

physical_devices = tf.config.experimental.list_physical_devices('GPU')
assert len(physical_devices) > 0, "Not enough GPU hardware devices available"
config = tf.config.experimental.set_memory_growth(physical_devices[0], True)

# Press the green button in the gutter to run the script.
if __name__ == '__main__':

    # ------------ task 1 "Data set" -----------------
    train, test, test_imgs = load_data()


    # ------------- task 2 "Model" -------------------
    discriminator = Discriminator()
    generator = Generator()

    # ------------- task 3 "Training" ----------------
    num_epochs = 10


    def visualization(train_losses, test_losses, name: str):
        plt.figure()
        line1, = plt.plot(train_losses)
        line2, = plt.plot(test_losses)
        plt.xlabel("Training steps")
        plt.ylabel(name)
        plt.legend((line1, line2), ("training", "test"))
        plt.show()

    training_d_losses = []
    test_d_losses = []
    training_g_losses = []
    test_g_losses = []

    for epoch in range(1, num_epochs + 1):
        print(f"Epoch {epoch}:")

        for data in train:
            metrics = train_step(discriminator, generator, data)

        print([f"{key}: {value}" for (key, value) in zip(list(metrics.keys()), list(metrics.values()))])
        training_d_losses.append(metrics['d_loss'])
        training_g_losses.append(metrics['g_loss'])

        discriminator.reset_metrics()
        generator.reset_metrics()

        for data in test:
            metrics = test_step(discriminator, generator, data)

        print([f"test_{key}: {value}" for (key, value) in zip(list(metrics.keys()), list(metrics.values()))])
        test_d_losses.append(metrics['d_loss'])
        test_g_losses.append(metrics['g_loss'])

        discriminator.reset_metrics()
        generator.reset_metrics()

        visualization(training_d_losses, test_d_losses, "Discriminator Loss")
        visualization(training_g_losses, test_g_losses, "Generator Loss")

        plt.figure(figsize=(20, 4))
        for i, inp in enumerate(test_imgs):
            n = 10
            random_input = tf.random.normal([1, 100])
            generated_img = generator(random_input)

            # Display originals
            ax = plt.subplot(2, n, i + 1)
            plt.imshow(tf.squeeze(inp))
            plt.gray()
            ax.get_xaxis().set_visible(False)
            ax.get_yaxis().set_visible(False)

            # Display generated
            ax = plt.subplot(2, n, i + 1 + n)
            plt.imshow(tf.squeeze(generated_img))
            plt.gray()
            ax.get_xaxis().set_visible(False)
            ax.get_yaxis().set_visible(False)

        plt.show()

