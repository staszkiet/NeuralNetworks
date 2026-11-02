"""
Końcowa sieć MNIST (trening: 60 000 obrazków, test: 10 000 obrazków).

Konfiguracja: [784, 128, 10], relu, softmax + CCE, lr = 0.05, batch = 64, 25 epok.
Dokładność train/test liczona po każdej epoce -> krzywe uczenia.
"""
import contextlib
import io
import os
from concurrent.futures import ProcessPoolExecutor
os.environ["OPENBLAS_NUM_THREADS"] = "10"
import matplotlib.pyplot as plt
import numpy as np
from main import MultiLayerPerceptron
from mnist import load_mnist
from vis_helpers import average_runs, pm, plot_curves, save_fig, save_runs

SEED = 42
N_RUNS = 10  # przebiegi n = 1..N_RUNS z ziarnem SEED * n
EPOCHS = 25

LAYER_DIMS = [784, 128, 10]
HIDDEN_ACTIVATION = "relu"
LEARNING_RATE = 0.05
BATCH_SIZE = 64

DATA_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "1")
X_train, Y_train = load_mnist(os.path.join(DATA_DIR, "train-images.idx3-ubyte"),
                              os.path.join(DATA_DIR, "train-labels.idx1-ubyte"))
X_test, Y_test = load_mnist(os.path.join(DATA_DIR, "t10k-images.idx3-ubyte"),
                            os.path.join(DATA_DIR, "t10k-labels.idx1-ubyte"))


def accuracy(model, X, Y):
    return np.mean(np.argmax(model.predict(X), axis=0) == np.argmax(Y, axis=0))


def train(seed):
    """Trening epoka po epoce (fit z epochs=1), żeby po każdej epoce zmierzyć dokładność train/test."""
    model = MultiLayerPerceptron(LAYER_DIMS, hidden_activation=HIDDEN_ACTIVATION, seed=seed)
    loss_curve, train_acc_curve, test_acc_curve = [], [], []
    for epoch in range(EPOCHS):
        with contextlib.redirect_stdout(io.StringIO()):
            history = model.fit(X_train, Y_train, epochs=1, learning_rate=LEARNING_RATE,
                                batch_size=BATCH_SIZE, task_type="multiclass_classification",
                                loss_func="cce")
        loss_curve.append(history["loss"][0])
        train_acc_curve.append(accuracy(model, X_train, Y_train))
        test_acc_curve.append(accuracy(model, X_test, Y_test))
        print(f"Epoch {epoch + 1}/{EPOCHS} | Strata (CCE): {loss_curve[-1]:.4f} | "
              f"dokł. train: {train_acc_curve[-1] * 100:.2f}% | dokł. test: {test_acc_curve[-1] * 100:.2f}%")

    test_loss = model.compute_loss(model.predict(X_test), Y_test)
    return {
        "seed": seed,
        "train_loss": loss_curve[-1],
        "test_loss": float(test_loss),
        "train_accuracy": train_acc_curve[-1],
        "accuracy": test_acc_curve[-1],
        "curve": np.array(loss_curve),
        "train_acc_curve": np.array(train_acc_curve),
        "test_acc_curve": np.array(test_acc_curve),
    }


def print_result(r):
    print(f"\n{'=' * 60}")
    print(f"MNIST | {LAYER_DIMS}, {HIDDEN_ACTIVATION}, lr = {LEARNING_RATE}, batch = {BATCH_SIZE}, {EPOCHS} epok")
    print(f"Średnia ± std z {N_RUNS} przebiegów (ziarna {SEED}·n)")
    print(f"{'=' * 60}")
    print(f"Loss train:      {pm(r, 'train_loss')}")
    print(f"Loss test:       {pm(r, 'test_loss')}")
    print(f"Dokł. train [%]: {pm(r, 'train_accuracy', 100, 2)}")
    print(f"Dokł. test [%]:  {pm(r, 'accuracy', 100, 2)}")


def plot_accuracy(r):
    epochs = np.arange(1, EPOCHS + 1)
    plt.figure(figsize=(9, 5))
    plt.plot(epochs, r["test_acc_curve"] * 100, color="#4C72B0", linewidth=1.8, label="test")
    plt.fill_between(epochs, (r["test_acc_curve"] - r["test_acc_curve_std"]) * 100,
                     (r["test_acc_curve"] + r["test_acc_curve_std"]) * 100, color="#4C72B0", alpha=0.15)
    plt.plot(epochs, r["train_acc_curve"] * 100, color="#4C72B0", linewidth=1, linestyle="--", label="train")
    plt.title(f"MNIST — końcowa sieć: dokładność po każdej epoce (średnia ± std z {N_RUNS} przebiegów)")
    plt.xlabel("Epoch")
    plt.ylabel("Dokładność [%]")
    plt.grid(True, linestyle="--", alpha=0.5)
    plt.legend()
    plt.tight_layout()


def main():
    with ProcessPoolExecutor(N_RUNS) as pool:
        runs = list(pool.map(train, [SEED * n for n in range(1, N_RUNS + 1)]))
    result = average_runs(runs)

    print_result(result)
    save_runs([{k: v for k, v in r.items() if not k.endswith("_acc_curve")} for r in runs], "mnist_final")
    plot_accuracy(result)
    save_fig("mnist_final")
    plot_curves(f"MNIST — końcowa sieć: strata treningowa ({N_RUNS} przebiegów)", [("-".join(map(str, LAYER_DIMS)), result)])
    save_fig("mnist_final_krzywe")
    plt.show()


if __name__ == "__main__":
    main()
