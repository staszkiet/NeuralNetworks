"""
Eksperyment: liczba warstw ukrytych (0–4) i ich szerokość.

layer_dims = [wejście, ukryte…, wyjście]; liczba warstw ukrytych = len(layer_dims) - 2.

SCENARIO: activation | cube | simple | three_gauss | all
"""
import os
import matplotlib.pyplot as plt
import numpy as np
from MLPRunner import MLPExperimentRunner
from vis_helpers import test_path as test_file_for, average_runs, pm, plot_curves, save_fig, save_runs

SCENARIO = "all"

SEED = 42
N_RUNS = 5  # przebiegi n = 1..N_RUNS z ziarnem SEED * n
EPOCHS = 300
LEARNING_RATE = 0.1
BATCH_SIZE = 32

base_dir = os.path.dirname(os.path.abspath(__file__))

SCENARIOS = {
    "activation": {
        "label": "Regresja (activation)",
        "train_file": os.path.join("regression", "data.activation.train.500.csv"),
        "input_dim": 1,
        "output_dim": 1,
        "loss_func": "mse",
        "hidden_activation": "relu",
    },
    "cube": {
        "label": "Regresja (cube)",
        "train_file": os.path.join("regression", "data.cube.train.500.csv"),
        "input_dim": 1,
        "output_dim": 1,
        "loss_func": "mse",
        "hidden_activation": "relu",
    },
    "simple": {
        "label": "Klasyfikacja binarna (simple)",
        "train_file": os.path.join("classification", "data.simple.train.1000.csv"),
        "input_dim": 2,
        "output_dim": 1,
        "loss_func": "bce",
        "hidden_activation": "tanh",
    },
    "three_gauss": {
        "label": "Klasyfikacja wieloklasowa (three_gauss)",
        "train_file": os.path.join("classification", "data.three_gauss.train.1000.csv"),
        "input_dim": 2,
        "output_dim": 3,
        "loss_func": "cce",
        "hidden_activation": "tanh",
    },
}

HIDDEN_WIDTH_SETS = {
    0: [[]],
    1: [[4], [16], [32]],
    2: [[8, 8], [16, 8], [32, 16]],
    3: [[8, 8, 8], [16, 12, 8], [24, 16, 8]],
    4: [[8, 8, 8, 8], [16, 12, 8, 4], [32, 24, 16, 8]],
}


def build_architectures(input_dim, output_dim):
    archs = []
    for n_hidden in range(0, 5):
        for widths in HIDDEN_WIDTH_SETS[n_hidden]:
            dims = [input_dim] + widths + [output_dim]
            if n_hidden == 0:
                name = "0 ukrytych (tylko wyjście)"
            else:
                name = f"{n_hidden} ukryte {widths}"
            archs.append({"n_hidden": n_hidden, "name": name, "layer_dims": dims})
    return archs


def run_architecture(runner, cfg, arch, seed):
    runner.configure_model(
        layer_dims=arch["layer_dims"],
        hidden_activation=cfg["hidden_activation"],
        loss_func=cfg["loss_func"],
        seed=seed,
    )
    history = runner.run_experiment(
        cfg["train_file"],
        epochs=EPOCHS,
        learning_rate=LEARNING_RATE,
        batch_size=BATCH_SIZE,
    )
    train_loss = history["loss"][-1]
    test_file = test_file_for(cfg["train_file"])
    if runner.task_type == "classification":
        test_loss, accuracy = runner.evaluate(test_file)
    else:
        test_loss = runner.evaluate(test_file)
        accuracy = None
    return {
        "train_loss": train_loss,
        "test_loss": test_loss,
        "accuracy": accuracy,
        "layer_dims": arch["layer_dims"],
        "curve": np.array(history["loss"]),
    }


def print_results(cfg, rows):
    is_clf = rows[0]["accuracy"] is not None
    print(f"\n{'=' * 88}")
    print(f"{cfg['label']} | aktywacja ukryta: {cfg['hidden_activation']}")
    print(f"{'=' * 88}")
    print(f"Średnia ± std z {N_RUNS} przebiegów (ziarna {SEED}·n)")
    if is_clf:
        print(f"{'# ukrytych':<10} {'Architektura':<28} {'layer_dims':<24} {'Loss test':>24} {'Acc test [%]':>16}")
    else:
        print(f"{'# ukrytych':<10} {'Architektura':<28} {'layer_dims':<24} {'Loss test':>24}")
    print("-" * 88)
    for row in rows:
        line = (
            f"{row['n_hidden']:<10} {row['name']:<28} {str(row['layer_dims']):<24} "
            f"{pm(row, 'test_loss'):>24}"
        )
        if is_clf:
            line += f" {pm(row, 'accuracy', 100, 2):>16}"
        print(line)


def plot_results(cfg, rows):
    is_clf = rows[0]["accuracy"] is not None
    metric = [(r["accuracy"] * 100 if is_clf else r["test_loss"]) for r in rows]
    metric_std = [(r["accuracy_std"] * 100 if is_clf else r["test_loss_std"]) for r in rows]
    metric_label = "Dokładność dla danych testowych [%]" if is_clf else f"Loss test ({cfg['loss_func'].upper()})"

    fig, ax = plt.subplots(figsize=(11, 5))
    x = np.arange(len(rows))
    colors = plt.cm.viridis([r["n_hidden"] / 4.0 for r in rows])
    ax.bar(x, metric, yerr=metric_std, capsize=3, color=colors)
    ax.set_xticks(x)
    ax.set_xticklabels([f"{r['n_hidden']}h\n{r['layer_dims']}" for r in rows], rotation=55, ha="right", fontsize=8)
    ax.set_ylabel(metric_label)
    ax.set_title(f"{cfg['label']} — głębokość i szerokość (średnia ± std z {N_RUNS} przebiegów)")
    ax.grid(True, axis="y", linestyle="--", alpha=0.5)
    fig.tight_layout()


def main():
    if SCENARIO == "all":
        keys = list(SCENARIOS.keys())
    elif SCENARIO in SCENARIOS:
        keys = [SCENARIO]

    runner = MLPExperimentRunner(data_dir=base_dir)

    for key in keys:
        cfg = SCENARIOS[key]
        train_path = os.path.join(base_dir, cfg["train_file"])
        test_full = os.path.join(base_dir, test_file_for(cfg["train_file"]))

        rows = []
        all_runs = []
        for arch in build_architectures(cfg["input_dim"], cfg["output_dim"]):
            runs = []
            for n in range(1, N_RUNS + 1):
                print(f"\n--- {cfg['label']} | {arch['name']} | {arch['layer_dims']} | przebieg {n} ---")
                runs.append({"n_hidden": arch["n_hidden"], "name": arch["name"], "seed": SEED * n,
                             **run_architecture(runner, cfg, arch, SEED * n)})
            rows.append(average_runs(runs))
            all_runs += runs

        print_results(cfg, rows)
        save_runs(all_runs, f"glebokosc_{key}")
        plot_results(cfg, rows)
        save_fig(f"glebokosc_{key}")
        plot_curves(f"{cfg['label']} — krzywe uczenia ({N_RUNS} przebiegów)",
                    [(f"{r['n_hidden']}h {r['layer_dims']}", r) for r in rows])
        save_fig(f"glebokosc_{key}_krzywe")
        plt.show()


if __name__ == "__main__":
    main()
