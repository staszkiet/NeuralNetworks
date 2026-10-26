"""Wizualizacja klasyfikacji wieloklasowej: data.three_gauss (2D)."""
import os
import numpy as np
import matplotlib.pyplot as plt
from MLPRunner import MLPExperimentRunner

base_dir = os.path.dirname(os.path.abspath(__file__))
train_file = os.path.join("classification", "data.three_gauss.train.1000.csv")

runner = MLPExperimentRunner(data_dir=base_dir)
if not os.path.exists(os.path.join(base_dir, train_file)):
    raise SystemExit(f"Brak pliku: {train_file}")

n_classes = 3
runner.configure_model(
    layer_dims=[2, 16, 8, n_classes],
    hidden_activation="tanh",
    loss_func="cce",
    seed=42,
)
runner.run_experiment(train_file, epochs=300, learning_rate=0.01, batch_size=32)

X_train, Y_train = runner.load_dataset(train_file)
y_labels = np.argmax(Y_train, axis=0)

x_min, x_max = X_train[0].min() - 0.5, X_train[0].max() + 0.5
y_min, y_max = X_train[1].min() - 0.5, X_train[1].max() + 0.5
xx, yy = np.meshgrid(np.linspace(x_min, x_max, 200), np.linspace(y_min, y_max, 200))
grid = np.c_[xx.ravel(), yy.ravel()].T
Z_class = np.argmax(runner.model.predict(grid), axis=0).reshape(xx.shape)

plt.figure(figsize=(8, 6))
plt.contourf(xx, yy, Z_class, alpha=0.35, cmap="RdYlBu")
plt.scatter(X_train[0], X_train[1], c=y_labels, cmap="RdYlBu", edgecolors="k", s=30)
plt.title(f"Klasyfikacja wieloklasowa (three_gauss)\n{train_file}")
plt.xlabel("X1")
plt.ylabel("X2")
plt.grid(True, linestyle="--", alpha=0.5)
plt.show()
