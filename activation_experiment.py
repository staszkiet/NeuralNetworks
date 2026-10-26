"""
Eksperyment: wpływ funkcji aktywacji w warstwach ukrytych (sigmoid, tanh, relu).

Aktywacja na wyjściu jest wybierana automatycznie przez model wg typu zadania:
  regresja          -> linear + MSE
  klasyf. binarna   -> sigmoid + BCE
  klasyf. wieloklas. -> softmax + CCE

SCENARIO: activation | cube | simple | three_gauss | all
"""
import os
import matplotlib.pyplot as plt
import numpy as np
from MLPRunner import MLPExperimentRunner
from vis_helpers import test_path as test_file_for, average_runs, pm, plot_curves, save_fig, save_runs

SCENARIO = "all"

HIDDEN_ACTIVATIONS = ["sigmoid", "tanh", "relu"]
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
        "layer_dims": [1, 12, 1],
        "loss_func": "mse",
        "output_note": "wyjście: linear",
    },
    "cube": {
        "label": "Regresja (cube)",
        "train_file": os.path.join("regression", "data.cube.train.500.csv"),
        "layer_dims": [1, 16, 16, 1],
        "loss_func": "mse",
        "output_note": "wyjście: linear",
    },
    "simple": {
        "label": "Klasyfikacja binarna (simple)",
        "train_file": os.path.join("classification", "data.simple.train.1000.csv"),
        "layer_dims": [2, 16, 8, 1],
        "loss_func": "bce",
        "output_note": "wyjście: sigmoid",
    },
    "three_gauss": {
        "label": "Klasyfikacja wieloklasowa (three_gauss)",
        "train_file": os.path.join("classification", "data.three_gauss.train.1000.csv"),
        "layer_dims": [2, 16, 8, 3],
        "loss_func": "cce",
        "output_note": "wyjście: softmax",
    },
}


def run_one(runner, cfg, hidden_activation, seed):
    runner.configure_model(
        layer_dims=cfg["layer_dims"],
        hidden_activation=hidden_activation,
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
    return {"train_loss": train_loss, "test_loss": test_loss, "accuracy": accuracy, "curve": np.array(history["loss"])}


def print_table(cfg, results):
    print(f"\n{'=' * 72}")
    print(f"{cfg['label']} | {cfg['output_note']}")
    print(f"Plik treningowy: {cfg['train_file']}")
    print(f"{'=' * 72}")
    print(f"Średnia ± std z {N_RUNS} przebiegów (ziarna {SEED}·n)")
    print(f"{'Aktywacja ukryta':<18} {'Loss train':>24} {'Loss test':>24} {'Dokł. test [%]':>16}")
    print("-" * 72)
    for act, row in results.items():
        acc_str = pm(row, "accuracy", 100, 2) if row["accuracy"] is not None else "—"
        print(
            f"{act:<18} {pm(row, 'train_loss'):>24} {pm(row, 'test_loss'):>24} {acc_str:>16}"
        )


def plot_scenario(cfg, results):
    acts = list(results.keys())
    test_losses = [results[a]["test_loss"] for a in acts]
    is_clf = results[acts[0]]["accuracy"] is not None

    ncols = 2 if is_clf else 1
    fig, axes = plt.subplots(1, ncols, figsize=(5 * ncols, 4))
    if ncols == 1:
        axes = [axes]

    axes[0].bar(acts, test_losses, yerr=[results[a]["test_loss_std"] for a in acts], capsize=4,
                color=["#4C72B0", "#55A868", "#C44E52"])
    axes[0].set_title(f"Strata testowa ({cfg['loss_func'].upper()})")
    axes[0].set_ylabel("Loss")
    axes[0].grid(True, axis="y", linestyle="--", alpha=0.5)

    if is_clf:
        accs = [results[a]["accuracy"] * 100 for a in acts]
        axes[1].bar(acts, accs, yerr=[results[a]["accuracy_std"] * 100 for a in acts], capsize=4,
                    color=["#4C72B0", "#55A868", "#C44E52"])
        axes[1].set_title("Dokładność dla danych testowych")
        axes[1].set_ylabel("Dokładność [%]")
        axes[1].set_ylim(0, 100)
        axes[1].grid(True, axis="y", linestyle="--", alpha=0.5)

    fig.suptitle(f"{cfg['label']}\nAktywacja ukryta | {cfg['output_note']} | średnia ± std z {N_RUNS} przebiegów")
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
        test_file = test_file_for(cfg["train_file"])
        test_full = os.path.join(base_dir, test_file)

        results = {}
        all_runs = []
        for act in HIDDEN_ACTIVATIONS:
            runs = []
            for n in range(1, N_RUNS + 1):
                print(f"\n--- {cfg['label']} | aktywacja ukryta: {act} | przebieg {n} ---")
                runs.append({"aktywacja": act, "seed": SEED * n, **run_one(runner, cfg, act, SEED * n)})
            results[act] = average_runs(runs)
            all_runs += runs

        print_table(cfg, results)
        save_runs(all_runs, f"aktywacje_{key}")
        plot_scenario(cfg, results)
        save_fig(f"aktywacje_{key}")
        plot_curves(f"{cfg['label']} — krzywe uczenia ({N_RUNS} przebiegów)", results.items())
        save_fig(f"aktywacje_{key}_krzywe")
        plt.show()


if __name__ == "__main__":
    main()
