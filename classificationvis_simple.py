"""Wizualizacja klasyfikacji binarnej: data.simple (2D) — train i test."""
import os
import numpy as np
import matplotlib.pyplot as plt
from MLPRunner import MLPExperimentRunner
from vis_helpers import test_path, xy_bounds, classification_grid


def labels_binary(Y):
    return Y.flatten().astype(int)


base_dir = os.path.dirname(os.path.abspath(__file__))
train_file = os.path.join("classification", "data.simple.train.1000.csv")
test_file = test_path(train_file)

runner = MLPExperimentRunner(data_dir=base_dir)
if not os.path.exists(os.path.join(base_dir, train_file)):
    raise SystemExit(f"Brak pliku: {train_file}")

runner.configure_model(layer_dims=[2, 16, 8, 1], hidden_activation="tanh", loss_func="bce", seed=42)
runner.run_experiment(train_file, epochs=300, learning_rate=0.01, batch_size=32)

X_train, Y_train = runner.load_dataset(train_file)
datasets = [(X_train, Y_train, f"Trening\n{train_file}")]

if os.path.exists(os.path.join(base_dir, test_file)):
    X_test, Y_test = runner.load_dataset(test_file)
    datasets.append((X_test, Y_test, f"Test\n{test_file}"))
    runner.evaluate(test_file)

x_min, x_max, y_min, y_max = xy_bounds(*[(X, Y) for X, Y, _ in datasets])
xx, yy, grid = classification_grid(x_min, x_max, y_min, y_max)
Z_bg = (runner.model.predict(grid)[0] >= 0.5).astype(int).reshape(xx.shape)

fig, axes = plt.subplots(1, len(datasets), figsize=(7 * len(datasets), 6), squeeze=False)
for ax, (X, Y, title) in zip(axes[0], datasets):
    ax.contourf(xx, yy, Z_bg, levels=[-0.5, 0.5, 1.5], cmap="RdYlBu", alpha=0.35)
    ax.scatter(
        X[0], X[1], c=labels_binary(Y), cmap="RdYlBu", vmin=0, vmax=1, edgecolors="k", s=30
    )
    ax.set_title(title)
    ax.set_xlabel("X1")
    ax.set_ylabel("X2")
    ax.grid(True, linestyle="--", alpha=0.5)

fig.suptitle("Klasyfikacja binarna (simple)")
fig.tight_layout()
plt.show()
