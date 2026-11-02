"""
Eksperyment MNIST: trening na zbiorze treningowym, test na zbiorze testowym.
"""
import os
import numpy as np
from main import MultiLayerPerceptron
from mnist import load_mnist

DATA_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "1")

EPOCHS = 10
LEARNING_RATE = 0.1
BATCH_SIZE = 64

X_train, Y_train = load_mnist(os.path.join(DATA_DIR, "train-images.idx3-ubyte"),
                              os.path.join(DATA_DIR, "train-labels.idx1-ubyte"))
X_test, Y_test = load_mnist(os.path.join(DATA_DIR, "t10k-images.idx3-ubyte"),
                            os.path.join(DATA_DIR, "t10k-labels.idx1-ubyte"))

model = MultiLayerPerceptron([784, 64, 10], hidden_activation='relu', seed=42)
model.fit(X_train, Y_train, epochs=EPOCHS, learning_rate=LEARNING_RATE, batch_size=BATCH_SIZE,
          task_type='multiclass_classification', loss_func='cce')

Y_pred = model.predict(X_test)
loss = model.compute_loss(Y_pred, Y_test)
accuracy = np.mean(np.argmax(Y_pred, axis=0) == np.argmax(Y_test, axis=0))
print(f"Test -> Strata (CCE): {loss:.4f} | Dokładność (Accuracy): {accuracy * 100:.2f}%")
