import sys
import pathlib
PROJECT_ROOT = pathlib.Path(__file__).resolve().parent.parent
sys.path.append(str(PROJECT_ROOT))

import os
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
plt.ioff()

from sklearn.model_selection import StratifiedKFold, LeaveOneGroupOut
from src.timaggregators.loaders import load_data
from src.timaggregators.features import build_features
from src.timaggregators.models import make_model_builders
from src.timaggregators.evaluation import get_probabilities

# Config
DATA_DIR = PROJECT_ROOT / "data" / "raw"
SCREENING_FILE = DATA_DIR / "screening_data.tsv"
DRUGS_FILE = DATA_DIR / "selected_drugs_smiles.tsv"
EXCIPIENTS_FILE = DATA_DIR / "selected_excipients_smiles.tsv"

OUTPUT_DIR = PROJECT_ROOT / "results" / "figures"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

DRUG_COL = "DRUG"
EXCIPIENT_COL = "EXCIPIENT"
LABEL_COL = "CLASS"

RANDOM_STATE = 42
N_SPLITS = 10
MODELS_TO_RUN = ["RandomForest", "LogisticRegression", "ExtraTrees"]

# Fold-wise prediction generators
def make_cv_predictions(X, y, builder, model_name):
    print(f"\nRunning CV predictions for {model_name}...")
    skf = StratifiedKFold(n_splits=N_SPLITS, shuffle=True, random_state=RANDOM_STATE)

    preds = np.zeros(len(X), dtype=float)

    for fold, (train_idx, test_idx) in enumerate(skf.split(X, y), start=1):
        print(f"  [CV] fold {fold}/{N_SPLITS}")

        X_train, X_test = X.iloc[train_idx], X.iloc[test_idx]
        y_train = y.iloc[train_idx]

        if y_train.nunique() < 2:
            preds[test_idx] = float(y_train.iloc[0])
            continue

        model = builder()
        model.fit(X_train, y_train)

        fold_probs = get_probabilities(model, X_test)
        preds[test_idx] = fold_probs

    return preds

def make_logo_predictions(X, y, groups, builder, model_name):
    print(f"\nRunning LOGO predictions for {model_name}...")
    logo = LeaveOneGroupOut()

    preds = np.zeros(len(X), dtype=float)
    total_folds = groups.nunique()

    for fold, (train_idx, test_idx) in enumerate(
        logo.split(X, y, groups=groups),
        start=1,
    ):
        held_out_drug = groups.iloc[test_idx].iloc[0]
        print(f"  [LOGO] fold {fold}/{total_folds} | held out: {held_out_drug}")

        X_train, X_test = X.iloc[train_idx], X.iloc[test_idx]
        y_train = y.iloc[train_idx]

        if y_train.nunique() < 2:
            preds[test_idx] = float(y_train.iloc[0])
            continue

        model = builder()
        model.fit(X_train, y_train)

        fold_probs = get_probabilities(model, X_test)
        preds[test_idx] = fold_probs

    return preds

# Plotting
def build_matrices(dataset, score_col, drug_order, excipient_order):
    screening_mat = dataset.pivot(
        index=DRUG_COL,
        columns=EXCIPIENT_COL,
        values="SCREENING_VALUE",
    ).reindex(index=drug_order, columns=excipient_order)

    pred_mat = dataset.pivot(
        index=DRUG_COL,
        columns=EXCIPIENT_COL,
        values=score_col,
    ).reindex(index=drug_order, columns=excipient_order)

    return screening_mat, pred_mat

def plot_heatmaps(
    screening_mat: pd.DataFrame,
    pred_mat: pd.DataFrame,
    model_name: str,
    evaluation_name: str,
    output_path: str,
):
    n_drugs = screening_mat.shape[0]
    n_excipients = screening_mat.shape[1]

    cell_size = 0.24
    fig_w = max(7, n_drugs * cell_size * 1.8)
    fig_h = max(14, n_excipients * cell_size * 1.15)

    fig, axes = plt.subplots(1, 2, figsize=(fig_w, fig_h), constrained_layout=True)

    im1 = axes[0].imshow(
        screening_mat.T.values,
        cmap="Greys",
        vmin=0,
        vmax=1,
        aspect="equal",
        interpolation="nearest",
    )
    axes[0].set_title("High-throughput screening")
    axes[0].set_xlabel("Drugs")
    axes[0].set_ylabel("Excipients")
    axes[0].set_xticks(np.arange(n_drugs))
    axes[0].set_xticklabels(screening_mat.index.tolist(), rotation=90, fontsize=7)
    axes[0].set_yticks(np.arange(n_excipients))
    axes[0].set_yticklabels(screening_mat.columns.tolist(), fontsize=7)
    cbar1 = fig.colorbar(im1, ax=axes[0], fraction=0.03, pad=0.02)
    cbar1.set_label("Screening hit (0/1)")

    im2 = axes[1].imshow(
        pred_mat.T.values,
        cmap="Greys",
        vmin=5,
        vmax=60,
        aspect="equal",
        interpolation="nearest",
    )
    axes[1].set_title(f"{evaluation_name} model prediction ({model_name})")
    axes[1].set_xlabel("Drugs")
    axes[1].set_ylabel("Excipients")
    axes[1].set_xticks(np.arange(n_drugs))
    axes[1].set_xticklabels(pred_mat.index.tolist(), rotation=90, fontsize=7)
    axes[1].set_yticks(np.arange(n_excipients))
    axes[1].set_yticklabels(pred_mat.columns.tolist(), fontsize=7)
    cbar2 = fig.colorbar(im2, ax=axes[1], fraction=0.03, pad=0.02)
    cbar2.set_label("Confidence (%)")

    plt.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close(fig)

    print(f"Saved: {output_path}")

def summarize_probabilities(y_true, y_prob):
    y_true = np.asarray(y_true, dtype=int)
    y_prob = np.asarray(y_prob, dtype=float)

    def one_row(split_name, arr):
        arr = np.asarray(arr, dtype=float)
        if len(arr) == 0:
            return {
                "split": split_name, "count": 0, "mean": np.nan, "std": np.nan,
                "min": np.nan, "p01": np.nan, "p05": np.nan, "p10": np.nan, "p25": np.nan,
                "p50": np.nan, "p75": np.nan, "p90": np.nan, "p95": np.nan, "p99": np.nan, "max": np.nan,
            }

        return {
            "split": split_name,
            "count": len(arr),
            "mean": float(np.mean(arr)),
            "std": float(np.std(arr)),
            "min": float(np.min(arr)),
            "p01": float(np.quantile(arr, 0.01)),
            "p05": float(np.quantile(arr, 0.05)),
            "p10": float(np.quantile(arr, 0.10)),
            "p25": float(np.quantile(arr, 0.25)),
            "p50": float(np.quantile(arr, 0.50)),
            "p75": float(np.quantile(arr, 0.75)),
            "p90": float(np.quantile(arr, 0.90)),
            "p95": float(np.quantile(arr, 0.95)),
            "p99": float(np.quantile(arr, 0.99)),
            "max": float(np.max(arr)),
        }

    rows = [
        one_row("all", y_prob),
        one_row("negative_class_0", y_prob[y_true == 0]),
        one_row("positive_class_1", y_prob[y_true == 1]),
    ]
    return pd.DataFrame(rows)

def plot_probability_distribution(
    y_true, y_prob, model_name, evaluation_name, output_path
):
    y_true = np.asarray(y_true, dtype=int)
    y_prob = np.asarray(y_prob, dtype=float)

    bins = np.linspace(0.0, 1.0, 41)

    fig, axes = plt.subplots(2, 1, figsize=(8, 8), constrained_layout=True)

    axes[0].hist(y_prob, bins=bins, edgecolor="black")
    axes[0].axvline(0.2, linestyle="--", linewidth=1, label="threshold = 0.2")
    axes[0].axvline(0.5, linestyle="--", linewidth=1, label="threshold = 0.5")
    axes[0].set_title(f"{evaluation_name} probability distribution ({model_name})")
    axes[0].set_xlabel("Predicted probability")
    axes[0].set_ylabel("Count")
    axes[0].set_xlim(0, 1)
    axes[0].legend()

    neg_probs = y_prob[y_true == 0]
    pos_probs = y_prob[y_true == 1]

    axes[1].hist(neg_probs, bins=bins, alpha=0.7, density=True, label=f"Class 0 (n={len(neg_probs)})", edgecolor="black")
    axes[1].hist(pos_probs, bins=bins, alpha=0.7, density=True, label=f"Class 1 (n={len(pos_probs)})", edgecolor="black")
    axes[1].axvline(0.2, linestyle="--", linewidth=1)
    axes[1].axvline(0.5, linestyle="--", linewidth=1)
    axes[1].set_title("Class-separated probability distribution")
    axes[1].set_xlabel("Predicted probability")
    axes[1].set_ylabel("Density")
    axes[1].set_xlim(0, 1)
    axes[1].legend()

    plt.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close(fig)

    print(f"Saved: {output_path}")

def main():
    screening_data, drugs_smiles, excipients_smiles = load_data(
        SCREENING_FILE, DRUGS_FILE, EXCIPIENTS_FILE, class_col=LABEL_COL
    )
    dataset = build_features(
        drugs_smiles=drugs_smiles,
        excipients_smiles=excipients_smiles,
        screening_data=screening_data,
    )

    dataset = dataset.copy()
    dataset["SCREENING_VALUE"] = dataset[LABEL_COL].astype(float)

    drug_order = [
        d for d in drugs_smiles["NAME"].tolist() if d in dataset[DRUG_COL].unique()
    ]
    excipient_order = [
        e for e in excipients_smiles["NAME"].tolist() if e in dataset[EXCIPIENT_COL].unique()
    ]

    builders = make_model_builders(random_state=RANDOM_STATE)

    for model_name in MODELS_TO_RUN:
        builder = builders[model_name]

        # CV predictions
        cv_preds = make_cv_predictions(X=dataset.drop(columns=[DRUG_COL, EXCIPIENT_COL, LABEL_COL]).copy(), y=dataset[LABEL_COL].astype(int).copy(), builder=builder, model_name=model_name)
        dataset_cv = dataset.copy()
        dataset_cv["PRED_PERCENT"] = cv_preds * 100.0

        screening_mat, pred_mat = build_matrices(
            dataset_cv,
            score_col="PRED_PERCENT",
            drug_order=drug_order,
            excipient_order=excipient_order,
        )

        cv_filename = OUTPUT_DIR / f"heatmap_cv_{model_name.lower()}.png"
        plot_heatmaps(
            screening_mat=screening_mat,
            pred_mat=pred_mat,
            model_name=model_name,
            evaluation_name="CV",
            output_path=str(cv_filename),
        )

        cv_dist_filename = OUTPUT_DIR / f"prob_dist_cv_{model_name.lower()}.png"
        plot_probability_distribution(
            y_true=dataset[LABEL_COL],
            y_prob=cv_preds,
            model_name=model_name,
            evaluation_name="CV",
            output_path=str(cv_dist_filename),
        )

        # LOGO predictions
        logo_preds = make_logo_predictions(X=dataset.drop(columns=[DRUG_COL, EXCIPIENT_COL, LABEL_COL]).copy(), y=dataset[LABEL_COL].astype(int).copy(), groups=dataset[DRUG_COL].copy(), builder=builder, model_name=model_name)
        dataset_logo = dataset.copy()
        dataset_logo["PRED_PERCENT"] = logo_preds * 100.0

        screening_mat, pred_mat = build_matrices(
            dataset_logo,
            score_col="PRED_PERCENT",
            drug_order=drug_order,
            excipient_order=excipient_order,
        )

        logo_filename = OUTPUT_DIR / f"heatmap_logo_{model_name.lower()}.png"
        plot_heatmaps(
            screening_mat=screening_mat,
            pred_mat=pred_mat,
            model_name=model_name,
            evaluation_name="LOGO",
            output_path=str(logo_filename),
        )

        logo_dist_filename = OUTPUT_DIR / f"prob_dist_logo_{model_name.lower()}.png"
        plot_probability_distribution(
            y_true=dataset[LABEL_COL],
            y_prob=logo_preds,
            model_name=model_name,
            evaluation_name="LOGO",
            output_path=str(logo_dist_filename),
        )

    print("\nDone.")
    print(f"All plots saved in: {OUTPUT_DIR}")
    plt.close("all")

if __name__ == "__main__":
    main()
