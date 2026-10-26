"""Wizualizacja regresji: data.activation (1D) — train i test."""
import os
import numpy as np
import matplotlib.pyplot as plt
from MLPRunner import MLPExperimentRunner
from vis_helpers import test_path


def plot_regression(ax, runner, X, Y, subtitle):
    sort_idx = np.argsort(X[0])
    X_sorted = X[:, sort_idx]
    Y_pred = runner.model.predict(X_sorted)
    ax.scatter(X[0], Y[0], color="black", label="Dane", alpha=0.5, s=15)
    ax.plot(X_sorted[0], Y_pred[0], color="red", linewidth=2.5, label="Model MLP")
    ax.set_title(subtitle)
    ax.set_xlabel("X")
    ax.set_ylabel("Y")
    ax.legend(loc="best", fontsize=8)
    ax.grid(True, linestyle="--", alpha=0.5)


base_dir = os.path.dirname(os.path.abspath(__file__))
train_file = os.path.join("regression", "data.activation.train.500.csv")
test_file = test_path(train_file)

runner = MLPExperimentRunner(data_dir=base_dir)
if not os.path.exists(os.path.join(base_dir, train_file)):
    raise SystemExit(f"Brak pliku: {train_file}")

runner.configure_model(layer_dims=[1, 10, 1], hidden_activation="relu", loss_func="mse", seed=42)
runner.run_experiment(train_file, epochs=200, learning_rate=0.01)

X_train, Y_train = runner.load_dataset(train_file)
panels = [(X_train, Y_train, f"Trening\n{train_file}")]

test_full = os.path.join(base_dir, test_file)
if os.path.exists(test_full):
    X_test, Y_test = runner.load_dataset(test_file)
    panels.append((X_test, Y_test, f"Test\n{test_file}"))
    runner.evaluate(test_file)

fig, axes = plt.subplots(1, len(panels), figsize=(7 * len(panels), 5), squeeze=False)
for ax, (X, Y, title) in zip(axes[0], panels):
    plot_regression(ax, runner, X, Y, title)
fig.suptitle("Regresja (activation)")
fig.tight_layout()
plt.show()
