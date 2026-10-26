"""Wizualizacja regresji: data.activation (1D)."""
import os
import numpy as np
import matplotlib.pyplot as plt
from MLPRunner import MLPExperimentRunner

base_dir = os.path.dirname(os.path.abspath(__file__))
train_file = os.path.join("regression", "data.activation.train.500.csv")

runner = MLPExperimentRunner(data_dir=base_dir)
if not os.path.exists(os.path.join(base_dir, train_file)):
    raise SystemExit(f"Brak pliku: {train_file}")

runner.configure_model(layer_dims=[1, 10, 1], hidden_activation="relu", loss_func="mse", seed=42)
runner.run_experiment(train_file, epochs=200, learning_rate=0.01)

X_train, Y_train = runner.load_dataset(train_file)
sort_idx = np.argsort(X_train[0])
X_sorted = X_train[:, sort_idx]
Y_pred = runner.model.predict(X_sorted)

plt.figure(figsize=(8, 5))
plt.scatter(X_train[0], Y_train[0], color="black", label="Zbiór uczący", alpha=0.5, s=15)
plt.plot(X_sorted[0], Y_pred[0], color="red", linewidth=2.5, label="Model MLP")
plt.title(f"Regresja (activation)\n{train_file}")
plt.xlabel("X")
plt.ylabel("Y")
plt.legend()
plt.grid(True, linestyle="--", alpha=0.5)
plt.show()
