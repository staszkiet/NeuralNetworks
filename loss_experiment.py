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
from vis_helpers import test_path as test_file_for

SCENARIO = "all"

SEED = 42
EPOCHS = 300
LEARNING_RATE = 0.01
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


def run_one(runner, cfg, loss_func):
    runner.configure_model(
        layer_dims=cfg["layer_dims"],
        hidden_activation=cfg["hidden_activation"],
        loss_func=loss_func,
        seed=SEED,
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
    }


def print_table(cfg, rows):
    is_clf = rows[0]["accuracy"] is not None
    print(f"\n{'=' * 80}")
    print(f"{cfg['label']}")
    print(f"{'=' * 80}")
    if is_clf:
        print(f"{'Loss uczenia':<12} {'Loss train':>12} {'Loss test':>12} {'Dokł. test':>14}")
        print("-" * 80)
        for r in rows:
            print(
                f"{r['loss_func'].upper():<12} {r['train_loss']:>12.4f} {r['test_loss']:>12.4f} "
                f"{r['accuracy'] * 100:>13.2f}%"
            )
    else:
        print(f"{'Loss uczenia':<12} {'Loss train':>12} {'MSE test':>12} {'MAE test':>12}")
        print("-" * 80)
        for r in rows:
            print(
                f"{r['loss_func'].upper():<12} {r['train_loss']:>12.4f} "
                f"{r['reg_mse']:>12.4f} {r['reg_mae']:>12.4f}"
            )


def plot_results(cfg, rows):
    is_clf = rows[0]["accuracy"] is not None
    labels = [r["loss_func"].upper() for r in rows]

    if is_clf:
        fig, axes = plt.subplots(1, 2, figsize=(10, 4))
        axes[0].bar(labels, [r["accuracy"] * 100 for r in rows], color=["#4C72B0", "#55A868"])
        axes[0].set_ylabel("Dokładność [%]")
        axes[0].set_ylim(0, 100)
        axes[0].set_title("Dokładność dla danych testowych")
        axes[0].grid(True, axis="y", linestyle="--", alpha=0.5)
        axes[1].bar(labels, [r["test_loss"] for r in rows], color=["#4C72B0", "#55A868"])
        axes[1].set_ylabel("Loss test")
        axes[1].set_title("Strata na teście")
        axes[1].grid(True, axis="y", linestyle="--", alpha=0.5)
    else:
        fig, ax = plt.subplots(figsize=(8, 4))
        x = np.arange(len(labels))
        w = 0.35
        ax.bar(x - w / 2, [r["reg_mse"] for r in rows], width=w, label="MSE test", color="#4C72B0")
        ax.bar(x + w / 2, [r["reg_mae"] for r in rows], width=w, label="MAE test", color="#55A868")
        ax.set_xticks(x)
        ax.set_xticklabels([f"uczenie: {l}" for l in labels])
        ax.set_ylabel("Wartość na zbiorze testowym")
        ax.set_title("MSE i MAE na teście")
        ax.legend()
        ax.grid(True, axis="y", linestyle="--", alpha=0.5)

    fig.suptitle(f"{cfg['label']} — porównanie funkcji straty")
    fig.tight_layout()
    plt.show()


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
        for loss_func in cfg["loss_funcs"]:
            print(f"\n--- {cfg['label']} | loss: {loss_func.upper()} ---")
            rows.append(run_one(runner, cfg, loss_func))

        print_table(cfg, rows)
        plot_results(cfg, rows)


if __name__ == "__main__":
    main()
