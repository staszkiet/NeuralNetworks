"""
Eksperymenty na MNIST (trening: 60 000 obrazków, test: 10 000 obrazków).

Bazowa konfiguracja: [784, 64, 10], relu, softmax + CCE, lr = 0.1, batch = 64.
W każdym eksperymencie zmieniamy jeden parametr, reszta zostaje bazowa.
Dokładność train/test liczona po każdej epoce -> krzywe uczenia.

EXPERIMENT: aktywacje | architektura | learning_rate | all
"""
import contextlib
import io
import os
import time
import matplotlib.pyplot as plt
import numpy as np
from main import MultiLayerPerceptron
from mnist import load_mnist
from vis_helpers import average_runs, pm, plot_curves, save_fig, save_runs

EXPERIMENT = "all"

SEED = 42
N_RUNS = 5  # przebiegi n = 1..N_RUNS z ziarnem SEED * n
EPOCHS = 25

BASE = {"layer_dims": [784, 64, 10], "hidden_activation": "relu", "learning_rate": 0.1, "batch_size": 64}

EXPERIMENTS = {
    "aktywacje": {
        "label": "Funkcja aktywacji w warstwie ukrytej",
        "param": "hidden_activation",
        "values": ["sigmoid", "tanh", "relu"],
    },
    "architektura": {
        "label": "Architektura sieci (liczba i rozmiar warstw ukrytych)",
        "param": "layer_dims",
        "values": [[784, 10], [784, 32, 10], [784, 64, 10], [784, 128, 10], [784, 64, 32, 10], [784, 128, 64, 10], [784, 256, 128, 10]],
    },
    "learning_rate": {
        "label": "Współczynnik uczenia",
        "param": "learning_rate",
        "values": [0.01, 0.02, 0.05, 0.1, 0.2, 0.5],
    },
}

DATA_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "1")
X_train, Y_train = load_mnist(os.path.join(DATA_DIR, "train-images.idx3-ubyte"),
                              os.path.join(DATA_DIR, "train-labels.idx1-ubyte"))
X_test, Y_test = load_mnist(os.path.join(DATA_DIR, "t10k-images.idx3-ubyte"),
                            os.path.join(DATA_DIR, "t10k-labels.idx1-ubyte"))


def value_label(value):
    if isinstance(value, list):
        return "-".join(map(str, value))
    return str(value)


def accuracy(model, X, Y):
    return np.mean(np.argmax(model.predict(X), axis=0) == np.argmax(Y, axis=0))


def train(cfg, seed):
    """Trening epoka po epoce (fit z epochs=1), żeby po każdej epoce zmierzyć dokładność train/test."""
    model = MultiLayerPerceptron(cfg["layer_dims"], hidden_activation=cfg["hidden_activation"], seed=seed)
    loss_curve, train_acc_curve, test_acc_curve = [], [], []
    start = time.time()
    for epoch in range(EPOCHS):
        with contextlib.redirect_stdout(io.StringIO()):
            history = model.fit(X_train, Y_train, epochs=1, learning_rate=cfg["learning_rate"],
                                batch_size=cfg["batch_size"], task_type="multiclass_classification",
                                loss_func="cce")
        loss_curve.append(history["loss"][0])
        train_acc_curve.append(accuracy(model, X_train, Y_train))
        test_acc_curve.append(accuracy(model, X_test, Y_test))
        print(f"Epoch {epoch + 1}/{EPOCHS} | Strata (CCE): {loss_curve[-1]:.4f} | "
              f"dokł. train: {train_acc_curve[-1] * 100:.2f}% | dokł. test: {test_acc_curve[-1] * 100:.2f}%")
    elapsed = time.time() - start

    test_loss = model.compute_loss(model.predict(X_test), Y_test)
    return model, {
        "train_loss": loss_curve[-1],
        "test_loss": float(test_loss),
        "train_accuracy": train_acc_curve[-1],
        "accuracy": test_acc_curve[-1],
        "czas_s": elapsed,
        "curve": np.array(loss_curve),
        "train_acc_curve": np.array(train_acc_curve),
        "test_acc_curve": np.array(test_acc_curve),
    }


def print_table(exp, results):
    print(f"\n{'=' * 96}")
    print(f"MNIST | {exp['label']}")
    print(f"Średnia ± std z {N_RUNS} przebiegów (ziarna {SEED}·n), {EPOCHS} epok")
    print(f"{'=' * 96}")
    print(f"{'Wariant':<16} {'Loss train':>18} {'Loss test':>18} {'Dokł. train [%]':>18} "
          f"{'Dokł. test [%]':>18} {'Czas [s]':>8}")
    print("-" * 96)
    for label, row in results.items():
        print(f"{label:<16} {pm(row, 'train_loss'):>18} {pm(row, 'test_loss'):>18} "
              f"{pm(row, 'train_accuracy', 100, 2):>18} {pm(row, 'accuracy', 100, 2):>18} {row['czas_s']:>8.1f}")


def plot_accuracy(exp, results):
    labels = list(results.keys())
    colors = plt.cm.tab10(np.arange(len(labels)))
    epochs = np.arange(1, EPOCHS + 1)
    fig, axes = plt.subplots(1, 2, figsize=(13, 4.5))

    for color, (label, r) in zip(colors, results.items()):
        axes[0].plot(epochs, r["test_acc_curve"] * 100, color=color, linewidth=1.8, label=f"{label} (test)")
        axes[0].fill_between(epochs, (r["test_acc_curve"] - r["test_acc_curve_std"]) * 100,
                             (r["test_acc_curve"] + r["test_acc_curve_std"]) * 100, color=color, alpha=0.15)
        axes[0].plot(epochs, r["train_acc_curve"] * 100, color=color, linewidth=1, linestyle="--",
                     label=f"{label} (train)")
    axes[0].set_title("Dokładność po każdej epoce (linia ciągła: test, przerywana: train)")
    axes[0].set_xlabel("Epoch")
    axes[0].set_ylabel("Dokładność [%]")
    axes[0].grid(True, linestyle="--", alpha=0.5)
    axes[0].legend(fontsize=7, ncol=2)

    x = np.arange(len(labels))
    width = 0.38
    axes[1].bar(x - width / 2, [results[l]["train_accuracy"] * 100 for l in labels], width,
                yerr=[results[l]["train_accuracy_std"] * 100 for l in labels], capsize=3,
                color="#8DA0CB", label="train")
    axes[1].bar(x + width / 2, [results[l]["accuracy"] * 100 for l in labels], width,
                yerr=[results[l]["accuracy_std"] * 100 for l in labels], capsize=3,
                color="#4C72B0", label="test")
    low = min(min(results[l]["accuracy"], results[l]["train_accuracy"]) for l in labels) * 100
    axes[1].set_ylim(max(0, low - 3), 100)
    axes[1].set_xticks(x)
    axes[1].set_xticklabels(labels, rotation=20)
    axes[1].set_title(f"Dokładność końcowa (po {EPOCHS} epokach)")
    axes[1].set_ylabel("Dokładność [%]")
    axes[1].grid(True, axis="y", linestyle="--", alpha=0.5)
    axes[1].legend()

    fig.suptitle(f"MNIST — {exp['label']} | średnia ± std z {N_RUNS} przebiegów")
    fig.tight_layout()


def run_experiment(key):
    exp = EXPERIMENTS[key]
    results = {}
    all_runs = []
    for value in exp["values"]:
        cfg = {**BASE, exp["param"]: value}
        label = value_label(value)
        runs = []
        for n in range(1, N_RUNS + 1):
            print(f"\n--- MNIST | {exp['param']} = {label} | przebieg {n} ---")
            _, row = train(cfg, SEED * n)
            runs.append({exp["param"]: label, "seed": SEED * n, **row})
        results[label] = average_runs(runs)
        all_runs += runs

    print_table(exp, results)
    save_runs([{k: v for k, v in r.items() if not k.endswith("_acc_curve")} for r in all_runs], f"mnist_{key}")
    plot_accuracy(exp, results)
    save_fig(f"mnist_{key}")
    plot_curves(f"MNIST — {exp['label']}: strata treningowa ({N_RUNS} przebiegów)", results.items())
    save_fig(f"mnist_{key}_krzywe")


def main():
    keys = list(EXPERIMENTS.keys()) if EXPERIMENT == "all" else [EXPERIMENT]
    for key in keys:
        run_experiment(key)
    plt.show()


if __name__ == "__main__":
    main()
