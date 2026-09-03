from pathlib import Path
import random

import numpy as np
import pandas as pd
import torch

from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline

from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier

from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    average_precision_score,
    matthews_corrcoef,
    confusion_matrix
)

from crushnet import (
    train_crushnet,
    predict_crushnet,
    CRUSHNet
)


# ============================================================
# CONFIGURATION
# ============================================================

RESULTS = Path("results")

EXPERIMENT_RESULTS = (
    RESULTS / "experiments"
)

CRUSH_RESULTS = (
    RESULTS / "crushnet"
)

EXPERIMENT_RESULTS.mkdir(
    parents=True,
    exist_ok=True
)

CRUSH_RESULTS.mkdir(
    parents=True,
    exist_ok=True
)

N_SAMPLES = 3000
N_FEATURES = 20
N_INFORMATIVE = 10
N_REDUNDANT = 4

TEST_SIZE = 0.20
VALIDATION_SIZE = 0.20

EPOCHS = 100
BATCH_SIZE = 64
LEARNING_RATE = 0.001

SEEDS = [42, 43, 44, 45, 46]

DEVICE = (
    "cuda"
    if torch.cuda.is_available()
    else "cpu"
)


# ============================================================
# REPRODUCIBILITY
# ============================================================

def set_seed(seed):

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


# ============================================================
# SYNTHETIC DATA GENERATOR
# ============================================================

def create_dataset(
    minority_ratio=0.10,
    class_sep=1.0,
    flip_y=0.01,
    seed=42
):

    majority_ratio = 1.0 - minority_ratio

    X, y = make_classification(
        n_samples=N_SAMPLES,
        n_features=N_FEATURES,
        n_informative=N_INFORMATIVE,
        n_redundant=N_REDUNDANT,
        n_repeated=0,
        n_classes=2,
        weights=[
            majority_ratio,
            minority_ratio
        ],
        class_sep=class_sep,
        flip_y=flip_y,
        random_state=seed
    )

    return X, y


# ============================================================
# DATA SPLIT + SCALING
# ============================================================

def prepare_data(
    X,
    y,
    seed
):

    X_train_full, X_test, y_train_full, y_test = (
        train_test_split(
            X,
            y,
            test_size=TEST_SIZE,
            stratify=y,
            random_state=seed
        )
    )

    X_train, X_val, y_train, y_val = (
        train_test_split(
            X_train_full,
            y_train_full,
            test_size=VALIDATION_SIZE,
            stratify=y_train_full,
            random_state=seed
        )
    )

    scaler = StandardScaler()

    X_train = scaler.fit_transform(
        X_train
    )

    X_val = scaler.transform(
        X_val
    )

    X_test = scaler.transform(
        X_test
    )

    return (
        X_train,
        X_val,
        X_test,
        y_train,
        y_val,
        y_test
    )


# ============================================================
# METRICS
# ============================================================

def calculate_metrics(
    y_true,
    y_pred,
    probabilities
):

    cm = confusion_matrix(
        y_true,
        y_pred,
        labels=[0, 1]
    )

    tn, fp, fn, tp = cm.ravel()

    try:
        roc_auc = roc_auc_score(
            y_true,
            probabilities[:, 1]
        )
    except ValueError:
        roc_auc = np.nan

    try:
        pr_auc = average_precision_score(
            y_true,
            probabilities[:, 1]
        )
    except ValueError:
        pr_auc = np.nan

    return {

        "accuracy": accuracy_score(
            y_true,
            y_pred
        ),

        "balanced_accuracy": balanced_accuracy_score(
            y_true,
            y_pred
        ),

        "precision": precision_score(
            y_true,
            y_pred,
            zero_division=0
        ),

        "recall": recall_score(
            y_true,
            y_pred,
            zero_division=0
        ),

        "macro_f1": f1_score(
            y_true,
            y_pred,
            average="macro",
            zero_division=0
        ),

        "minority_f1": f1_score(
            y_true,
            y_pred,
            pos_label=1,
            zero_division=0
        ),

        "minority_recall": recall_score(
            y_true,
            y_pred,
            pos_label=1,
            zero_division=0
        ),

        "roc_auc": roc_auc,

        "pr_auc": pr_auc,

        "mcc": matthews_corrcoef(
            y_true,
            y_pred
        ),

        "tn": tn,
        "fp": fp,
        "fn": fn,
        "tp": tp
    }


# ============================================================
# BASELINE MODELS
# ============================================================

def get_baselines():

    return {

        "Logistic Regression":
            Pipeline([
                (
                    "scaler",
                    StandardScaler()
                ),
                (
                    "model",
                    LogisticRegression(
                        max_iter=2000
                    )
                )
            ]),

        "Balanced Logistic":
            Pipeline([
                (
                    "scaler",
                    StandardScaler()
                ),
                (
                    "model",
                    LogisticRegression(
                        class_weight="balanced",
                        max_iter=2000
                    )
                )
            ]),

        "Random Forest":
            RandomForestClassifier(
                n_estimators=300,
                random_state=42,
                n_jobs=-1
            ),

        "Balanced Random Forest":
            RandomForestClassifier(
                n_estimators=300,
                class_weight="balanced",
                random_state=42,
                n_jobs=-1
            ),

        "kNN":
            Pipeline([
                (
                    "scaler",
                    StandardScaler()
                ),
                (
                    "model",
                    KNeighborsClassifier(
                        n_neighbors=15
                    )
                )
            ])
    }


# ============================================================
# CRUSH-NET EXPERIMENT
# ============================================================

def run_crushnet(
    X_train,
    X_val,
    X_test,
    y_train,
    y_val,
    y_test,
    seed,
    save_history=False
):

    history_path = None

    if save_history:

        history_path = (
            CRUSH_RESULTS /
            "training_history.csv"
        )

    model, history = train_crushnet(

        X_train,
        y_train,

        X_val,
        y_val,

        input_dim=X_train.shape[1],

        epochs=EPOCHS,

        batch_size=BATCH_SIZE,

        learning_rate=LEARNING_RATE,

        device=DEVICE,

        seed=seed,

        save_path=history_path,

        verbose=False
    )

    predictions, probabilities, gate = (
        predict_crushnet(
            model,
            X_test,
            DEVICE
        )
    )

    metrics = calculate_metrics(
        y_test,
        predictions,
        probabilities
    )

    metrics["gate_mean"] = np.mean(
        gate
    )

    metrics["gate_min"] = np.min(
        gate
    )

    metrics["gate_max"] = np.max(
        gate
    )

    return metrics


# ============================================================
# BASELINE COMPARISON
# ============================================================

def run_baseline_comparison():

    rows = []

    conditions = [
        (0.10, 1.0, 0.01),
        (0.05, 1.0, 0.01),
        (0.02, 1.0, 0.01),
        (0.01, 1.0, 0.01)
    ]

    for minority_ratio, class_sep, flip_y in conditions:

        print(
            f"\nBaseline condition: "
            f"{minority_ratio:.0%} minority"
        )

        for seed in SEEDS:

            set_seed(seed)

            X, y = create_dataset(
                minority_ratio=minority_ratio,
                class_sep=class_sep,
                flip_y=flip_y,
                seed=seed
            )

            (
                X_train,
                X_val,
                X_test,
                y_train,
                y_val,
                y_test
            ) = prepare_data(
                X,
                y,
                seed
            )

            baselines = get_baselines()

            for name, model in baselines.items():

                model.fit(
                    X_train,
                    y_train
                )

                predictions = model.predict(
                    X_test
                )

                if hasattr(
                    model,
                    "predict_proba"
                ):

                    probabilities = (
                        model.predict_proba(
                            X_test
                        )
                    )

                else:

                    probabilities = np.zeros(
                        (
                            len(X_test),
                            2
                        )
                    )

                metrics = calculate_metrics(
                    y_test,
                    predictions,
                    probabilities
                )

                row = {

                    "model": name,

                    "minority_ratio":
                        minority_ratio,

                    "class_sep":
                        class_sep,

                    "flip_y":
                        flip_y,

                    "seed":
                        seed
                }

                row.update(
                    metrics
                )

                rows.append(
                    row
                )

            # CRUSH-Net

            metrics = run_crushnet(
                X_train,
                X_val,
                X_test,
                y_train,
                y_val,
                y_test,
                seed,
                save_history=(
                    seed == SEEDS[0]
                    and minority_ratio == 0.10
                )
            )

            row = {

                "model": "CRUSH-Net",

                "minority_ratio":
                    minority_ratio,

                "class_sep":
                    class_sep,

                "flip_y":
                    flip_y,

                "seed":
                    seed
            }

            row.update(
                metrics
            )

            rows.append(
                row
            )

    df = pd.DataFrame(
        rows
    )

    output = (
        EXPERIMENT_RESULTS /
        "baseline_comparison.csv"
    )

    df.to_csv(
        output,
        index=False
    )

    print(
        f"\nSaved: {output}"
    )

    return df


# ============================================================
# IMBALANCE ROBUSTNESS
# ============================================================

def run_imbalance_experiment():

    rows = []

    ratios = [
        0.10,
        0.05,
        0.02,
        0.01
    ]

    for ratio in ratios:

        print(
            f"\nImbalance: "
            f"{ratio:.0%} minority"
        )

        for seed in SEEDS:

            set_seed(seed)

            X, y = create_dataset(
                minority_ratio=ratio,
                class_sep=1.0,
                flip_y=0.01,
                seed=seed
            )

            (
                X_train,
                X_val,
                X_test,
                y_train,
                y_val,
                y_test
            ) = prepare_data(
                X,
                y,
                seed
            )

            metrics = run_crushnet(
                X_train,
                X_val,
                X_test,
                y_train,
                y_val,
                y_test,
                seed
            )

            row = {

                "model": "CRUSH-Net",

                "minority_ratio":
                    ratio,

                "seed":
                    seed
            }

            row.update(
                metrics
            )

            rows.append(
                row
            )

    df = pd.DataFrame(
        rows
    )

    df.to_csv(
        EXPERIMENT_RESULTS /
        "imbalance_robustness.csv",
        index=False
    )

    summary = (
        df.groupby(
            "minority_ratio"
        )[
            [
                "balanced_accuracy",
                "macro_f1",
                "minority_f1",
                "minority_recall",
                "roc_auc",
                "pr_auc",
                "mcc"
            ]
        ]
        .agg(
            ["mean", "std"]
        )
    )

    summary.to_csv(
        EXPERIMENT_RESULTS /
        "imbalance_summary.csv"
    )

    return df


# ============================================================
# DIFFICULTY / NOISE EXPERIMENT
# ============================================================

def run_difficulty_experiment():

    rows = []

    conditions = [

        {
            "condition": "Easy",
            "class_sep": 2.0,
            "flip_y": 0.00
        },

        {
            "condition": "Moderate",
            "class_sep": 1.5,
            "flip_y": 0.01
        },

        {
            "condition": "Difficult",
            "class_sep": 1.0,
            "flip_y": 0.03
        },

        {
            "condition": "Very Difficult",
            "class_sep": 0.5,
            "flip_y": 0.05
        }
    ]

    for condition in conditions:

        print(
            f"\nDifficulty: "
            f"{condition['condition']}"
        )

        for seed in SEEDS:

            set_seed(seed)

            X, y = create_dataset(
                minority_ratio=0.05,
                class_sep=condition["class_sep"],
                flip_y=condition["flip_y"],
                seed=seed
            )

            (
                X_train,
                X_val,
                X_test,
                y_train,
                y_val,
                y_test
            ) = prepare_data(
                X,
                y,
                seed
            )

            metrics = run_crushnet(
                X_train,
                X_val,
                X_test,
                y_train,
                y_val,
                y_test,
                seed
            )

            row = {

                "condition":
                    condition["condition"],

                "class_sep":
                    condition["class_sep"],

                "flip_y":
                    condition["flip_y"],

                "minority_ratio":
                    0.05,

                "seed":
                    seed
            }

            row.update(
                metrics
            )

            rows.append(
                row
            )

    df = pd.DataFrame(
        rows
    )

    df.to_csv(
        EXPERIMENT_RESULTS /
        "difficulty_robustness.csv",
        index=False
    )

    return df


# ============================================================
# ABLATION MODEL
# ============================================================

class AblationCRUSHNet(CRUSHNet):

    def __init__(
        self,
        input_dim,
        remove_router=False,
        remove_attenuation=False,
        remove_clones=False
    ):

        super().__init__(
            input_dim=input_dim
        )

        self.remove_router = remove_router
        self.remove_attenuation = remove_attenuation
        self.remove_clones = remove_clones

    def encode(self, x):

        z_major = self.encoder_major(x)
        z_minor = self.encoder_minor(x)

        if self.remove_router:

            gate = torch.full(
                (x.shape[0], 1),
                0.5,
                device=x.device
            )

        else:

            gate = self.router(x)

        z = (
            (1.0 - gate) * z_major
            + gate * z_minor
        )

        if self.remove_attenuation:

            pass

        else:

            z = self.attn(
                z,
                x
            )

        return z, gate


def train_ablation_model(
    variant,
    X_train,
    y_train,
    X_val,
    y_val,
    seed
):

    set_seed(seed)

    remove_router = (
        variant == "Without Router"
    )

    remove_attenuation = (
        variant == "Without Attenuation"
    )

    remove_clones = (
        variant == "Without Pseudo-Clones"
    )

    model = AblationCRUSHNet(
        input_dim=X_train.shape[1],
        remove_router=remove_router,
        remove_attenuation=remove_attenuation,
        remove_clones=remove_clones
    ).to(DEVICE)

    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=LEARNING_RATE
    )

    X_train_t = torch.tensor(
        X_train,
        dtype=torch.float32
    ).to(DEVICE)

    y_train_t = torch.tensor(
        y_train,
        dtype=torch.long
    ).to(DEVICE)

    X_val_t = torch.tensor(
        X_val,
        dtype=torch.float32
    ).to(DEVICE)

    y_val_t = torch.tensor(
        y_val,
        dtype=torch.long
    ).to(DEVICE)

    dataset = torch.utils.data.TensorDataset(
        X_train_t,
        y_train_t
    )

    loader = torch.utils.data.DataLoader(
        dataset,
        batch_size=BATCH_SIZE,
        shuffle=True
    )

    best_state = None
    best_val = float("inf")
    patience_counter = 0

    for epoch in range(EPOCHS):

        model.train()

        for xb, yb in loader:

            output = model(
                xb,
                yb,
                generate_clones=not remove_clones
            )

            main_loss = F.cross_entropy(
                output["logits"],
                yb
            )

            router_loss = torch.tensor(
                0.0,
                device=DEVICE
            )

            if not remove_router:

                router_loss = F.binary_cross_entropy(
                    output["gate"],
                    yb.float().unsqueeze(1)
                )

            clone_loss = torch.tensor(
                0.0,
                device=DEVICE
            )

            if (
                not remove_clones
                and output.get("clone_logits") is not None
            ):

                clone_loss = F.cross_entropy(
                    output["clone_logits"],
                    output["clone_targets"]
                )

            loss = (
                main_loss
                + clone_loss
                + 0.1 * router_loss
            )

            optimizer.zero_grad()

            loss.backward()

            optimizer.step()

        model.eval()

        with torch.no_grad():

            val_output = model(
                X_val_t
            )

            val_loss = F.cross_entropy(
                val_output["logits"],
                y_val_t
            ).item()

        if val_loss < best_val:

            best_val = val_loss

            best_state = {
                key: value.detach().cpu().clone()
                for key, value
                in model.state_dict().items()
            }

            patience_counter = 0

        else:

            patience_counter += 1

        if patience_counter >= 12:

            break

    if best_state is not None:

        model.load_state_dict(
            best_state
        )

    return model


# ============================================================
# ABLATION EXPERIMENT
# ============================================================

def run_ablation():

    variants = [
        "Full CRUSH-Net",
        "Without Router",
        "Without Attenuation",
        "Without Pseudo-Clones"
    ]

    rows = []

    for variant in variants:

        print(
            f"\nAblation: {variant}"
        )

        for seed in SEEDS:

            X, y = create_dataset(
                minority_ratio=0.05,
                class_sep=1.0,
                flip_y=0.01,
                seed=seed
            )

            (
                X_train,
                X_val,
                X_test,
                y_train,
                y_val,
                y_test
            ) = prepare_data(
                X,
                y,
                seed
            )

            if variant == "Full CRUSH-Net":

                metrics = run_crushnet(
                    X_train,
                    X_val,
                    X_test,
                    y_train,
                    y_val,
                    y_test,
                    seed
                )

            else:

                model = train_ablation_model(
                    variant,
                    X_train,
                    y_train,
                    X_val,
                    y_val,
                    seed
                )

                predictions, probabilities, gate = (
                    predict_crushnet(
                        model,
                        X_test,
                        DEVICE
                    )
                )

                metrics = calculate_metrics(
                    y_test,
                    predictions,
                    probabilities
                )

            row = {

                "variant":
                    variant,

                "seed":
                    seed
            }

            row.update(
                metrics
            )

            rows.append(
                row
            )

    df = pd.DataFrame(
        rows
    )

    df.to_csv(
        EXPERIMENT_RESULTS /
        "ablation_results.csv",
        index=False
    )

    summary = (
        df.groupby(
            "variant"
        )[
            [
                "balanced_accuracy",
                "macro_f1",
                "minority_f1",
                "minority_recall",
                "roc_auc",
                "pr_auc",
                "mcc"
            ]
        ]
        .agg(
            ["mean", "std"]
        )
    )

    summary.to_csv(
        EXPERIMENT_RESULTS /
        "ablation_summary.csv"
    )

    return df


# ============================================================
# MAIN
# ============================================================

if __name__ == "__main__":

    print("=" * 60)
    print("CRUSH-Net EXPERIMENTAL FRAMEWORK")
    print("=" * 60)

    print(
        f"\nDevice: {DEVICE}"
    )

    print(
        "\n[1/4] Baseline comparison..."
    )

    run_baseline_comparison()

    print(
        "\n[2/4] Imbalance robustness..."
    )

    run_imbalance_experiment()

    print(
        "\n[3/4] Difficulty/noise robustness..."
    )

    run_difficulty_experiment()

    print(
        "\n[4/4] Ablation study..."
    )

    run_ablation()

    print(
        "\n" + "=" * 60
    )

    print(
        "ALL EXPERIMENTS COMPLETE"
    )

    print(
        "=" * 60
    )

    print(
        f"\nResults saved to:\n"
        f"{RESULTS.resolve()}"
    )