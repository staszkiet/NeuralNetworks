"""
Eksperyment: wpływ miary błędu (loss) na skuteczność uczenia.

Regresja: MSE vs MAE (wyjście linear)
Binarna: BCE vs MSE (wyjście sigmoid)
Wieloklasowa: CCE vs MSE (wyjście softmax)

SCENARIO: activation | cube | simple | three_gauss | all
"""
import os
import matplotlib.pyplot as plt
import numpy as np
from MLPRunner import MLPExperimentRunner
from main import LossFunctions
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
        "layer_dims": [1, 12, 1],
        "hidden_activation": "relu",
        "loss_funcs": ["mse", "mae"],
    },
    "cube": {
        "label": "Regresja (cube)",
        "train_file": os.path.join("regression", "data.cube.train.500.csv"),
        "layer_dims": [1, 16, 16, 1],
        "hidden_activation": "relu",
        "loss_funcs": ["mse", "mae"],
    },
    "simple": {
        "label": "Klasyfikacja binarna (simple)",
        "train_file": os.path.join("classification", "data.simple.train.1000.csv"),
        "layer_dims": [2, 16, 8, 1],
        "hidden_activation": "tanh",
        "loss_funcs": ["bce", "mse"],
    },
    "three_gauss": {
        "label": "Klasyfikacja wieloklasowa (three_gauss)",
        "train_file": os.path.join("classification", "data.three_gauss.train.1000.csv"),
        "layer_dims": [2, 16, 8, 3],
        "hidden_activation": "tanh",
        "loss_funcs": ["cce", "mse"],
    },
}


def regression_test_scores(runner, test_file):
    X_test, Y_test = runner.load_dataset(test_file)
    Y_pred = runner.model.predict(X_test)
    return {
        "mse": LossFunctions.mse(Y_pred, Y_test),
        "mae": LossFunctions.mae(Y_pred, Y_test),
    }


def run_one(runner, cfg, loss_func, seed):
    runner.configure_model(
        layer_dims=cfg["layer_dims"],
        hidden_activation=cfg["hidden_activation"],
        loss_func=loss_func,
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
        return {
            "loss_func": loss_func,
            "train_loss": train_loss,
            "test_loss": test_loss,
            "accuracy": accuracy,
            "reg_mse": None,
            "reg_mae": None,
            "curve": np.array(history["loss"]),
        }

    scores = regression_test_scores(runner, test_file)
    test_loss = runner.evaluate(test_file)
    return {
        "loss_func": loss_func,
        "train_loss": train_loss,
        "test_loss": test_loss,
        "accuracy": None,
        "reg_mse": scores["mse"],
        "reg_mae": scores["mae"],
        "curve": np.array(history["loss"]),
    }


def print_table(cfg, rows):
    is_clf = rows[0]["accuracy"] is not None
    print(f"\n{'=' * 80}")
    print(f"{cfg['label']}")
    print(f"{'=' * 80}")
    print(f"Średnia ± std z {N_RUNS} przebiegów (ziarna {SEED}·n)")
    if is_clf:
        print(f"{'Loss uczenia':<12} {'Loss train':>24} {'Loss test':>24} {'Dokł. test [%]':>16}")
        print("-" * 80)
        for r in rows:
            print(
                f"{r['loss_func'].upper():<12} {pm(r, 'train_loss'):>24} {pm(r, 'test_loss'):>24} "
                f"{pm(r, 'accuracy', 100, 2):>16}"
            )
    else:
        print(f"{'Loss uczenia':<12} {'Loss train':>24} {'MSE test':>24} {'MAE test':>24}")
        print("-" * 80)
        for r in rows:
            print(
                f"{r['loss_func'].upper():<12} {pm(r, 'train_loss'):>24} "
                f"{pm(r, 'reg_mse'):>24} {pm(r, 'reg_mae'):>24}"
            )


def plot_results(cfg, rows):
    is_clf = rows[0]["accuracy"] is not None
    labels = [r["loss_func"].upper() for r in rows]

    if is_clf:
        fig, axes = plt.subplots(1, 2, figsize=(10, 4))
        axes[0].bar(labels, [r["accuracy"] * 100 for r in rows], yerr=[r["accuracy_std"] * 100 for r in rows],
                    capsize=4, color=["#4C72B0", "#55A868"])
        axes[0].set_ylabel("Dokładność [%]")
        axes[0].set_ylim(0, 100)
        axes[0].set_title("Dokładność dla danych testowych")
        axes[0].grid(True, axis="y", linestyle="--", alpha=0.5)
        axes[1].bar(labels, [r["test_loss"] for r in rows], yerr=[r["test_loss_std"] for r in rows],
                    capsize=4, color=["#4C72B0", "#55A868"])
        axes[1].set_ylabel("Loss test")
        axes[1].set_title("Strata na teście")
        axes[1].grid(True, axis="y", linestyle="--", alpha=0.5)
    else:
        fig, ax = plt.subplots(figsize=(8, 4))
        x = np.arange(len(labels))
        w = 0.35
        ax.bar(x - w / 2, [r["reg_mse"] for r in rows], width=w, yerr=[r["reg_mse_std"] for r in rows],
               capsize=4, label="MSE test", color="#4C72B0")
        ax.bar(x + w / 2, [r["reg_mae"] for r in rows], width=w, yerr=[r["reg_mae_std"] for r in rows],
               capsize=4, label="MAE test", color="#55A868")
        ax.set_xticks(x)
        ax.set_xticklabels([f"uczenie: {l}" for l in labels])
        ax.set_ylabel("Wartość na zbiorze testowym")
        ax.set_title("MSE i MAE na teście")
        ax.legend()
        ax.grid(True, axis="y", linestyle="--", alpha=0.5)

    fig.suptitle(f"{cfg['label']} — porównanie funkcji straty (średnia ± std z {N_RUNS} przebiegów)")
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
        for loss_func in cfg["loss_funcs"]:
            runs = []
            for n in range(1, N_RUNS + 1):
                print(f"\n--- {cfg['label']} | loss: {loss_func.upper()} | przebieg {n} ---")
                runs.append({"seed": SEED * n, **run_one(runner, cfg, loss_func, SEED * n)})
            rows.append(average_runs(runs))
            all_runs += runs

        print_table(cfg, rows)
        save_runs(all_runs, f"strata_{key}")
        plot_results(cfg, rows)
        save_fig(f"strata_{key}")
        # każda krzywa jest w swojej mierze błędu — porównujemy tempo zbieżności, nie wartości
        plot_curves(f"{cfg['label']} — krzywe uczenia ({N_RUNS} przebiegów)",
                    [(r["loss_func"].upper(), r) for r in rows])
        save_fig(f"strata_{key}_krzywe")
        plt.show()


if __name__ == "__main__":
    main()
