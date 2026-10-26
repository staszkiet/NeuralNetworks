"""
Wizualizacja wag w kolejnych epokach (history['weights'] z fit()).
Ustaw TASK jak w errorvis.py.
"""
import os
import numpy as np
import matplotlib.pyplot as plt
from MLPRunner import MLPExperimentRunner

TASK = "simple"  # activation | cube | simple | three_gauss

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


def snapshot_epoch_indices(n_epochs):
    if n_epochs <= 1:
        return [0]
    candidates = [0, n_epochs // 4, n_epochs // 2, (3 * n_epochs) // 4, n_epochs - 1]
    return sorted(set(candidates))

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
    learning_rate=0.01,
    batch_size=32,
)

weight_history = history["weights"]
n_layers = len(weight_history[0])
snap_idxs = snapshot_epoch_indices(len(weight_history))

# --- Wykres 1: norma Frobeniusa wag w każdej warstwie vs epoch ---
plt.figure(figsize=(9, 5))
epochs_x = range(1, len(weight_history) + 1)
for layer_i in range(n_layers):
    norms = [np.linalg.norm(weight_history[e][layer_i], "fro") for e in range(len(weight_history))]
    plt.plot(epochs_x, norms, linewidth=2, label=f"Warstwa {layer_i + 1} (W)")
plt.title(f"Norma wag w epokach — {cfg['task_label']}")
plt.xlabel("Epoka")
plt.ylabel("||W||_F")
plt.grid(True, linestyle="--", alpha=0.5)
plt.legend()
plt.tight_layout()
plt.show()

# --- Wykres 2: mapy ciepła W w wybranych epokach (wiersz = warstwa) ---
n_snaps = len(snap_idxs)
fig, axes = plt.subplots(n_layers, n_snaps, figsize=(2.2 * n_snaps, 2.5 * n_layers), squeeze=False)

for layer_i in range(n_layers):
    mats = [weight_history[ei][layer_i] for ei in snap_idxs]
    vmax = max(np.max(np.abs(m)) for m in mats)
    vmax = vmax if vmax > 0 else 1.0

    for col, (ei, W) in enumerate(zip(snap_idxs, mats)):
        ax = axes[layer_i, col]
        im = ax.imshow(W, aspect="auto", cmap="RdBu_r", vmin=-vmax, vmax=vmax)
        ax.set_title(f"ep. {ei + 1}", fontsize=9)
        ax.set_xlabel("wejście")
        if col == 0:
            ax.set_ylabel(f"W warstwy {layer_i + 1}\nneuron")

fig.suptitle(f"Macierze wag — {cfg['task_label']}\n{cfg['train_file']}", y=1.02)
fig.tight_layout()
cbar_ax = fig.add_axes([0.92, 0.15, 0.02, 0.7])
fig.colorbar(im, cax=cbar_ax, label="waga")
plt.show()
