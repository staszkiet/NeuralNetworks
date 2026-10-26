"""
Krzywa straty względem epoch. Ustaw TASK na jeden z: activation, cube, simple, three_gauss.
"""
import os
import matplotlib.pyplot as plt
from MLPRunner import MLPExperimentRunner

TASK = "three_gauss"  # activation | cube | simple | three_gauss

base_dir = os.path.dirname(os.path.abspath(__file__))

EXPERIMENTS = {
    "activation": {
        "train_file": os.path.join("regression", "data.activation.train.500.csv"),
        "layer_dims": [1, 12, 1],
        "hidden_activation": "relu",
        "loss_func": "mse",
        "task_label": "regresja (activation)",
        "epochs": 300,
    },
    "cube": {
        "train_file": os.path.join("regression", "data.cube.train.500.csv"),
        "layer_dims": [1, 16, 16, 1],
        "hidden_activation": "relu",
        "loss_func": "mse",
        "task_label": "regresja (cube)",
        "epochs": 300,
    },
    "simple": {
        "train_file": os.path.join("classification", "data.simple.train.1000.csv"),
        "layer_dims": [2, 16, 8, 1],
        "hidden_activation": "tanh",
        "loss_func": "bce",
        "task_label": "klasyfikacja binarna (simple)",
        "epochs": 300,
    },
    "three_gauss": {
        "train_file": os.path.join("classification", "data.three_gauss.train.1000.csv"),
        "layer_dims": [2, 16, 8, 3],
        "hidden_activation": "tanh",
        "loss_func": "cce",
        "task_label": "klasyfikacja wieloklasowa (three_gauss)",
        "epochs": 300,
    },
}


cfg = EXPERIMENTS[TASK]
train_path = os.path.join(base_dir, cfg["train_file"])

runner = MLPExperimentRunner(data_dir=base_dir)
runner.configure_model(
    layer_dims=cfg["layer_dims"],
    hidden_activation=cfg["hidden_activation"],
    loss_func=cfg["loss_func"],
    seed=42,
)
history = runner.run_experiment(
    cfg["train_file"],
    epochs=cfg["epochs"],
    learning_rate=0.1,
    batch_size=32,
)

losses = history["loss"]
epochs_x = range(1, len(losses) + 1)

plt.figure(figsize=(8, 5))
plt.plot(epochs_x, losses, color="purple", linewidth=2, label=f"Loss ({cfg['loss_func'].upper()})")
plt.title(f"Krzywa uczenia — {cfg['task_label']}\n{cfg['train_file']}")
plt.xlabel("Epoch")
plt.ylabel("Strata")
plt.grid(True, linestyle="--", alpha=0.5)
plt.legend()
plt.show()
