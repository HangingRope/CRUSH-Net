from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


RESULTS = Path("results")

FIGURES = RESULTS / "figures"

FIGURES.mkdir(
    parents=True,
    exist_ok=True
)


# ============================================================
# TRAINING CURVE
# ============================================================

def plot_training():

    file = (
        RESULTS /
        "crushnet" /
        "training_history.csv"
    )

    if not file.exists():
        print("Training history not found.")
        return

    df = pd.read_csv(file)

    plt.figure(figsize=(8, 5))

    plt.plot(
        df["epoch"],
        df["loss"],
        label="Total Loss"
    )

    plt.plot(
        df["epoch"],
        df["main_loss"],
        label="Classification Loss"
    )

    plt.plot(
        df["epoch"],
        df["clone_loss"],
        label="Pseudo-Clone Loss"
    )

    plt.plot(
        df["epoch"],
        df["router_loss"],
        label="Router Loss"
    )

    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.title("CRUSH-Net Training Dynamics")

    plt.legend()
    plt.grid(alpha=0.25)
    plt.tight_layout()

    plt.savefig(
        FIGURES / "training_curve.png",
        dpi=300
    )

    plt.close()


# ============================================================
# BASELINE COMPARISON
# ============================================================

def plot_baselines():

    file = (
        RESULTS /
        "experiments" /
        "baseline_comparison.csv"
    )

    if not file.exists():
        print("Baseline results not found.")
        return

    df = pd.read_csv(file)

    summary = (
        df.groupby("model")["minority_f1"]
        .agg(["mean", "std"])
        .sort_values("mean")
    )

    plt.figure(figsize=(9, 6))

    plt.barh(
        summary.index,
        summary["mean"],
        xerr=summary["std"],
        capsize=4
    )

    plt.xlabel("Minority-Class F1")
    plt.ylabel("Model")
    plt.title(
        "Baseline Comparison of Minority-Class F1"
    )

    plt.grid(
        axis="x",
        alpha=0.25
    )

    plt.tight_layout()

    plt.savefig(
        FIGURES / "baseline_comparison.png",
        dpi=300
    )

    plt.close()


# ============================================================
# IMBALANCE ROBUSTNESS
# ============================================================

def plot_imbalance():

    file = (
        RESULTS /
        "experiments" /
        "imbalance_robustness.csv"
    )

    if not file.exists():
        print("Imbalance results not found.")
        return

    df = pd.read_csv(file)

    summary = (
        df.groupby("minority_ratio")[
            "minority_f1"
        ]
        .agg(["mean", "std"])
        .reset_index()
    )

    summary = summary.sort_values(
        "minority_ratio"
    )

    x = (
        summary["minority_ratio"] * 100
    )

    plt.figure(figsize=(8, 5))

    plt.errorbar(
        x,
        summary["mean"],
        yerr=summary["std"],
        marker="o",
        capsize=4,
        linewidth=2
    )

    plt.xlabel(
        "Minority-Class Proportion (%)"
    )

    plt.ylabel(
        "Minority-Class F1"
    )

    plt.title(
        "CRUSH-Net Robustness to Class Imbalance"
    )

    plt.grid(alpha=0.25)

    plt.tight_layout()

    plt.savefig(
        FIGURES / "imbalance_robustness.png",
        dpi=300
    )

    plt.close()


# ============================================================
# DIFFICULTY ROBUSTNESS
# ============================================================

def plot_difficulty():

    file = (
        RESULTS /
        "experiments" /
        "difficulty_robustness.csv"
    )

    if not file.exists():
        print("Difficulty results not found.")
        return

    df = pd.read_csv(file)

    order = [
        "Easy",
        "Moderate",
        "Difficult",
        "Very Difficult"
    ]

    summary = (
        df.groupby("condition")[
            "minority_f1"
        ]
        .agg(["mean", "std"])
        .reindex(order)
    )

    plt.figure(figsize=(9, 5))

    x = np.arange(
        len(summary)
    )

    plt.errorbar(
        x,
        summary["mean"],
        yerr=summary["std"],
        marker="o",
        capsize=4,
        linewidth=2
    )

    plt.xticks(
        x,
        summary.index
    )

    plt.xlabel(
        "Classification Difficulty"
    )

    plt.ylabel(
        "Minority-Class F1"
    )

    plt.title(
        "CRUSH-Net Robustness to Classification Difficulty"
    )

    plt.grid(alpha=0.25)

    plt.tight_layout()

    plt.savefig(
        FIGURES / "difficulty_robustness.png",
        dpi=300
    )

    plt.close()


# ============================================================
# ABLATION
# ============================================================

def plot_ablation():

    file = (
        RESULTS /
        "experiments" /
        "ablation_results.csv"
    )

    if not file.exists():
        print("Ablation results not found.")
        return

    df = pd.read_csv(file)

    summary = (
        df.groupby("variant")[
            "minority_f1"
        ]
        .agg(["mean", "std"])
        .sort_values("mean")
    )

    plt.figure(figsize=(9, 6))

    plt.barh(
        summary.index,
        summary["mean"],
        xerr=summary["std"],
        capsize=4
    )

    plt.xlabel(
        "Minority-Class F1"
    )

    plt.ylabel(
        "Architecture"
    )

    plt.title(
        "CRUSH-Net Ablation Study"
    )

    plt.grid(
        axis="x",
        alpha=0.25
    )

    plt.tight_layout()

    plt.savefig(
        FIGURES / "ablation.png",
        dpi=300
    )

    plt.close()


# ============================================================
# CONFUSION MATRIX
# ============================================================

def plot_confusion_matrix():

    file = (
        RESULTS /
        "experiments" /
        "baseline_comparison.csv"
    )

    if not file.exists():
        return

    df = pd.read_csv(file)

    crush = df[
        df["model"] == "CRUSH-Net"
    ]

    if crush.empty:
        return

    cm = np.array([
        [
            crush["tn"].mean(),
            crush["fp"].mean()
        ],
        [
            crush["fn"].mean(),
            crush["tp"].mean()
        ]
    ])

    plt.figure(figsize=(6, 5))

    plt.imshow(cm)

    plt.colorbar()

    plt.xticks(
        [0, 1],
        ["Predicted 0", "Predicted 1"]
    )

    plt.yticks(
        [0, 1],
        ["True 0", "True 1"]
    )

    for i in range(2):
        for j in range(2):

            plt.text(
                j,
                i,
                f"{cm[i, j]:.1f}",
                ha="center",
                va="center"
            )

    plt.xlabel("Predicted")
    plt.ylabel("True")

    plt.title(
        "Mean CRUSH-Net Confusion Matrix"
    )

    plt.tight_layout()

    plt.savefig(
        FIGURES / "confusion_matrix.png",
        dpi=300
    )

    plt.close()


# ============================================================
# BASELINE TABLE
# ============================================================

def create_baseline_table():

    file = (
        RESULTS /
        "experiments" /
        "baseline_comparison.csv"
    )

    if not file.exists():
        return

    df = pd.read_csv(file)

    columns = [
        "balanced_accuracy",
        "macro_f1",
        "minority_f1",
        "minority_recall",
        "roc_auc",
        "pr_auc",
        "mcc"
    ]

    table = (
        df.groupby("model")[columns]
        .agg(["mean", "std"])
        .round(4)
    )

    table.to_csv(
        FIGURES /
        "table_baseline_results.csv"
    )


# ============================================================
# ABLATION TABLE
# ============================================================

def create_ablation_table():

    file = (
        RESULTS /
        "experiments" /
        "ablation_results.csv"
    )

    if not file.exists():
        return

    df = pd.read_csv(file)

    columns = [
        "balanced_accuracy",
        "macro_f1",
        "minority_f1",
        "minority_recall",
        "roc_auc",
        "pr_auc",
        "mcc"
    ]

    table = (
        df.groupby("variant")[columns]
        .agg(["mean", "std"])
        .round(4)
    )

    table.to_csv(
        FIGURES /
        "table_ablation_results.csv"
    )


# ============================================================
# DIFFICULTY TABLE
# ============================================================

def create_difficulty_table():

    file = (
        RESULTS /
        "experiments" /
        "difficulty_robustness.csv"
    )

    if not file.exists():
        return

    df = pd.read_csv(file)

    columns = [
        "balanced_accuracy",
        "macro_f1",
        "minority_f1",
        "minority_recall",
        "roc_auc",
        "pr_auc",
        "mcc"
    ]

    table = (
        df.groupby("condition")[columns]
        .agg(["mean", "std"])
        .round(4)
    )

    table.to_csv(
        FIGURES /
        "table_difficulty_results.csv"
    )


# ============================================================
# MAIN
# ============================================================

if __name__ == "__main__":

    print(
        "\nGenerating CRUSH-Net figures...\n"
    )

    plot_training()

    plot_baselines()

    plot_imbalance()

    plot_difficulty()

    plot_ablation()

    plot_confusion_matrix()

    create_baseline_table()

    create_ablation_table()

    create_difficulty_table()

    print(
        "\nFigures and tables saved to:"
    )

    print(
        FIGURES.resolve()
    )