import matplotlib.pyplot as plt
import numpy as np
from scipy import spatial


def visualization(train_losses, test_losses):
    plt.figure()
    (line1,) = plt.plot(train_losses)
    (line2,) = plt.plot(test_losses)
    plt.xlabel("Training steps")
    plt.ylabel("Loss")
    plt.legend((line1, line2), ("training", "test"))
    plt.show()


def print_closest_word(skipgram, test_words, word_to_id, id_to_word):
    embedding_matrix = skipgram.embedding_weights.numpy()

    for word in test_words:
        i = word_to_id[word]

        closest = np.argmin(
            [spatial.distance.cosine(embedding_matrix[i], embedding_matrix[q])
             if q != i else np.inf for q in range(len(embedding_matrix))])

        print(f"Closest to '{word}' is '{id_to_word[closest]}'")
