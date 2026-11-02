"""
Wczytywanie MNIST z plików IDX (np. train-images.idx3-ubyte + train-labels.idx1-ubyte).
"""
import numpy as np


def load_mnist(images_file, labels_file):
    images = np.fromfile(images_file, dtype=np.uint8, offset=16)  # pomijamy 16-bajtowy nagłówek
    labels = np.fromfile(labels_file, dtype=np.uint8, offset=8)   # pomijamy 8-bajtowy nagłówek

    X = images.reshape(-1, 784).T / 255.0  # Kształt: (784, n_samples), piksele 0..1
    Y = np.eye(10)[labels].T               # Kształt: (10, n_samples), one-hot
    return X, Y
