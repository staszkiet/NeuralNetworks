"""Wspólne fragmenty wykresów train/test (wizualizacje)."""
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

OUT_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "wyniki")


def out_path(filename):
    os.makedirs(OUT_DIR, exist_ok=True)
    return os.path.join(OUT_DIR, filename)


def save_fig(name):
    """Zapisuje bieżący wykres do wyniki/<name>.png."""
    plt.savefig(out_path(name + ".png"), dpi=150, bbox_inches="tight")


def save_runs(runs, name):
    """Wyniki wszystkich przebiegów (bez krzywych uczenia) do wyniki/<name>.csv."""
    pd.DataFrame(runs).drop(columns="curve").to_csv(out_path(name + ".csv"), index=False)


def average_runs(runs):
    """Średnia i std (klucz + '_std') pól liczbowych z przebiegów o różnych ziarnach."""
    out = dict(runs[0])
    for key, val in runs[0].items():
        if isinstance(val, (float, np.ndarray)):
            vals = np.array([r[key] for r in runs], dtype=float)
            out[key] = vals.mean(axis=0)
            out[key + "_std"] = vals.std(axis=0)
    return out


def pm(row, key, scale=1.0, digits=4):
    """'średnia ± std' dla pola uśrednionego przez average_runs."""
    return f"{row[key] * scale:.{digits}f} ± {row[key + '_std'] * scale:.{digits}f}"


def plot_curves(title, labeled_rows):
    """Uśrednione krzywe uczenia porównywanych wariantów na jednym wykresie."""
    plt.figure(figsize=(9, 5))
    for label, r in labeled_rows:
        plt.plot(np.arange(1, len(r["curve"]) + 1), r["curve"], linewidth=1.5, label=label)
    plt.yscale("log")
    plt.title(title)
    plt.xlabel("Epoch")
    plt.ylabel("Strata treningowa (średnia z przebiegów)")
    plt.grid(True, linestyle="--", alpha=0.5)
    plt.legend(fontsize=7, ncol=2)
    plt.tight_layout()


def test_path(train_file):
    return train_file.replace(".train.", ".test.")


def xy_bounds(*datasets, pad=0.5):
    """datasets: pary (X, Y) z load_dataset; X ma kształt (2, N)."""
    xs = np.hstack([X[0] for X, _ in datasets])
    ys = np.hstack([X[1] for X, _ in datasets])
    return xs.min() - pad, xs.max() + pad, ys.min() - pad, ys.max() + pad


def classification_grid(x_min, x_max, y_min, y_max, n=200):
    xx, yy = np.meshgrid(np.linspace(x_min, x_max, n), np.linspace(y_min, y_max, n))
    grid = np.c_[xx.ravel(), yy.ravel()].T
    return xx, yy, grid
