"""Wspólne fragmenty wykresów train/test (wizualizacje)."""
import numpy as np


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
