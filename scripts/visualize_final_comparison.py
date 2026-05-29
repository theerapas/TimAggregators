import argparse
import pathlib
import sys

PROJECT_ROOT = pathlib.Path(__file__).resolve().parent.parent
sys.path.append(str(PROJECT_ROOT))

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    average_precision_score,
    precision_recall_curve,
    roc_auc_score,
    roc_curve,
)
from sklearn.model_selection import LeaveOneGroupOut

from src.timaggregators.evaluation import get_probabilities
from src.timaggregators.features import build_features
from src.timaggregators.loaders import load_data


DATA_DIR = PROJECT_ROOT / "data" / "raw"
SCREENING_FILE = DATA_DIR / "screening_data.tsv"
DRUGS_FILE = DATA_DIR / "selected_drugs_smiles.tsv"
EXCIPIENTS_FILE = DATA_DIR / "selected_excipients_smiles.tsv"

OLD_INFERENCE_FILE = PROJECT_ROOT / "results" / "inference" / "all_pair_scores.csv"
NEW_INFERENCE_FILE = (
    PROJECT_ROOT / "results" / "final_inference" / "improved_all_pair_scores.csv"
)
OUT_DIR = PROJECT_ROOT / "results" / "final_validation_visualizations"

RANDOM_STATE = 42
N_ESTIMATORS = 500


def parse_args():
    parser = argparse.ArgumentParser(
        description=(
            "Create final comparison plots using actual HTS labels and LOGO-only "
            "validation predictions."
        )
    )
    parser.add_argument("--output-dir", type=pathlib.Path, default=OUT_DIR)
    parser.add_argument("--old-inference-file", type=pathlib.Path, default=OLD_INFERENCE_FILE)
    parser.add_argument("--new-inference-file", type=pathlib.Path, default=NEW_INFERENCE_FILE)
    parser.add_argument("--scatter-sample", type=int, default=100_000)
    parser.add_argument("--skip-logo", action="store_true")
    parser.add_argument(
        "--include-inference-diagnostics",
        action="store_true",
        help=(
            "Also write old/new inference score summaries. These are not validation "
            "metrics because the inference pool is unlabeled."
        ),
    )
    return parser.parse_args()


def make_rf(class_weight=None):
    return RandomForestClassifier(
        n_estimators=N_ESTIMATORS,
        random_state=RANDOM_STATE,
        n_jobs=-1,
        class_weight=class_weight,
    )


def make_logo_predictions(dataset, components, model_name, class_weight=None):
    print(f"\nRunning LOGO predictions: {model_name}")
    X = dataset.drop(columns=["DRUG", "EXCIPIENT", "CLASS"]).copy()
    X = X.replace([np.inf, -np.inf], np.nan).fillna(0.0)
    y = dataset["CLASS"].astype(int).copy()
    groups = dataset["DRUG"].copy()

    logo = LeaveOneGroupOut()
    preds = np.zeros(len(X), dtype=float)
    total_folds = groups.nunique()

    for fold, (train_idx, test_idx) in enumerate(logo.split(X, y, groups=groups), start=1):
        held_out = groups.iloc[test_idx].iloc[0]
        print(f"  [{model_name}] LOGO fold {fold}/{total_folds}: {held_out}")

        model = make_rf(class_weight=class_weight)
        model.fit(X.iloc[train_idx], y.iloc[train_idx])
        preds[test_idx] = get_probabilities(model, X.iloc[test_idx])

    output = dataset[["DRUG", "EXCIPIENT", "CLASS"]].copy()
    output[f"{model_name}_LOGO_PROB"] = preds
    output[f"{model_name}_FEATURES"] = "+".join(components)
    return output


def ordered_labels(screening_data, drugs_smiles, excipients_smiles):
    drug_order = [
        name
        for name in drugs_smiles["NAME"].astype(str).tolist()
        if name in set(screening_data["DRUG"].astype(str))
    ]
    excipient_order = [
        name
        for name in excipients_smiles["NAME"].astype(str).tolist()
        if name in set(screening_data["EXCIPIENT"].astype(str))
    ]
    return drug_order, excipient_order


def pivot_matrix(df, value_col, drug_order, excipient_order):
    return (
        df.pivot(index="DRUG", columns="EXCIPIENT", values=value_col)
        .reindex(index=drug_order, columns=excipient_order)
    )


def plot_combined_old_style_heatmap(
    comparison,
    output_path,
    drug_order,
    excipient_order,
):
    actual = pivot_matrix(comparison, "CLASS", drug_order, excipient_order)
    old_percent = pivot_matrix(
        comparison.assign(PRED_PERCENT=comparison["OLD_BASELINE_LOGO_PROB"] * 100.0),
        "PRED_PERCENT",
        drug_order,
        excipient_order,
    )
    new_percent = pivot_matrix(
        comparison.assign(PRED_PERCENT=comparison["NEW_IMPROVED_LOGO_PROB"] * 100.0),
        "PRED_PERCENT",
        drug_order,
        excipient_order,
    )

    n_drugs = len(drug_order)
    n_excipients = len(excipient_order)
    cell_size = 0.24
    fig_w = max(10.5, n_drugs * cell_size * 2.7)
    fig_h = max(14, n_excipients * cell_size * 1.15)

    fig, axes = plt.subplots(1, 3, figsize=(fig_w, fig_h), constrained_layout=True)

    im1 = axes[0].imshow(
        actual.T.values,
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
    axes[0].set_xticklabels(actual.index.tolist(), rotation=90, fontsize=7)
    axes[0].set_yticks(np.arange(n_excipients))
    axes[0].set_yticklabels(actual.columns.tolist(), fontsize=7)
    cbar1 = fig.colorbar(im1, ax=axes[0], fraction=0.03, pad=0.02)
    cbar1.set_label("Screening hit (0/1)")

    im2 = axes[1].imshow(
        old_percent.T.values,
        cmap="Greys",
        vmin=5,
        vmax=60,
        aspect="equal",
        interpolation="nearest",
    )
    axes[1].set_title("LOGO prediction (old baseline)")
    axes[1].set_xlabel("Drugs")
    axes[1].set_ylabel("Excipients")
    axes[1].set_xticks(np.arange(n_drugs))
    axes[1].set_xticklabels(old_percent.index.tolist(), rotation=90, fontsize=7)
    axes[1].set_yticks(np.arange(n_excipients))
    axes[1].set_yticklabels(old_percent.columns.tolist(), fontsize=7)
    cbar2 = fig.colorbar(im2, ax=axes[1], fraction=0.03, pad=0.02)
    cbar2.set_label("Confidence (%)")

    im3 = axes[2].imshow(
        new_percent.T.values,
        cmap="Greys",
        vmin=5,
        vmax=60,
        aspect="equal",
        interpolation="nearest",
    )
    axes[2].set_title("LOGO prediction (new improved)")
    axes[2].set_xlabel("Drugs")
    axes[2].set_ylabel("Excipients")
    axes[2].set_xticks(np.arange(n_drugs))
    axes[2].set_xticklabels(new_percent.index.tolist(), rotation=90, fontsize=7)
    axes[2].set_yticks(np.arange(n_excipients))
    axes[2].set_yticklabels(new_percent.columns.tolist(), fontsize=7)
    cbar3 = fig.colorbar(im3, ax=axes[2], fraction=0.03, pad=0.02)
    cbar3.set_label("Confidence (%)")

    fig.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved: {output_path}")


def plot_logo_probability_distribution(comparison, output_path):
    y = comparison["CLASS"].astype(int).to_numpy()
    old_prob = comparison["OLD_BASELINE_LOGO_PROB"].astype(float).to_numpy()
    new_prob = comparison["NEW_IMPROVED_LOGO_PROB"].astype(float).to_numpy()
    bins = np.linspace(0, 1, 41)

    fig, axes = plt.subplots(3, 1, figsize=(8, 11), constrained_layout=True)

    axes[0].hist(old_prob, bins=bins, alpha=0.65, label="Old baseline LOGO", edgecolor="black")
    axes[0].hist(new_prob, bins=bins, alpha=0.65, label="New improved LOGO", edgecolor="black")
    axes[0].axvline(0.2, linestyle="--", linewidth=1, label="old threshold 0.20")
    axes[0].axvline(0.3, linestyle="--", linewidth=1, label="new threshold 0.30")
    axes[0].set_title("LOGO probability distribution, all HTS pairs")
    axes[0].set_xlabel("Predicted probability")
    axes[0].set_ylabel("Count")
    axes[0].set_xlim(0, 1)
    axes[0].legend()

    axes[1].hist(
        old_prob[y == 0],
        bins=bins,
        alpha=0.65,
        density=True,
        label="Old baseline, class 0",
        edgecolor="black",
    )
    axes[1].hist(
        new_prob[y == 0],
        bins=bins,
        alpha=0.65,
        density=True,
        label="New improved, class 0",
        edgecolor="black",
    )
    axes[1].set_title("LOGO probabilities for actual negatives")
    axes[1].set_xlabel("Predicted probability")
    axes[1].set_ylabel("Density")
    axes[1].set_xlim(0, 1)
    axes[1].legend()

    axes[2].hist(
        old_prob[y == 1],
        bins=bins,
        alpha=0.65,
        density=True,
        label="Old baseline, class 1",
        edgecolor="black",
    )
    axes[2].hist(
        new_prob[y == 1],
        bins=bins,
        alpha=0.65,
        density=True,
        label="New improved, class 1",
        edgecolor="black",
    )
    axes[2].set_title("LOGO probabilities for actual positives")
    axes[2].set_xlabel("Predicted probability")
    axes[2].set_ylabel("Density")
    axes[2].set_xlim(0, 1)
    axes[2].legend()

    fig.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved: {output_path}")


def enrichment_curve(y_true, y_prob):
    order = np.argsort(-np.asarray(y_prob, dtype=float))
    y_sorted = np.asarray(y_true, dtype=int)[order]
    ranks = np.arange(1, len(y_sorted) + 1)
    cumulative_hits = np.cumsum(y_sorted)
    positives = max(int(np.sum(y_sorted)), 1)
    baseline_rate = positives / len(y_sorted)
    recall_at_rank = cumulative_hits / positives
    precision_at_rank = cumulative_hits / ranks
    lift_at_rank = precision_at_rank / baseline_rate
    screened_fraction = ranks / len(y_sorted)
    return screened_fraction, recall_at_rank, precision_at_rank, lift_at_rank


def plot_logo_validation_curves(comparison, output_path):
    y_true = comparison["CLASS"].astype(int).to_numpy()
    models = [
        ("Old baseline LOGO", comparison["OLD_BASELINE_LOGO_PROB"].astype(float).to_numpy()),
        ("New improved LOGO", comparison["NEW_IMPROVED_LOGO_PROB"].astype(float).to_numpy()),
    ]

    fig, axes = plt.subplots(1, 3, figsize=(16, 5), constrained_layout=True)

    for label, probs in models:
        precision, recall, _ = precision_recall_curve(y_true, probs)
        auprc = average_precision_score(y_true, probs)
        axes[0].plot(recall, precision, label=f"{label} AUPRC={auprc:.3f}")

        fpr, tpr, _ = roc_curve(y_true, probs)
        auroc = roc_auc_score(y_true, probs)
        axes[1].plot(fpr, tpr, label=f"{label} AUROC={auroc:.3f}")

        screened_fraction, recall_at_rank, _, lift_at_rank = enrichment_curve(
            y_true, probs
        )
        axes[2].plot(
            screened_fraction * 100,
            recall_at_rank * 100,
            label=f"{label}",
        )

    positive_rate = float(np.mean(y_true))
    axes[0].axhline(positive_rate, linestyle="--", linewidth=1, color="black")
    axes[0].set_title("Precision-recall on LOGO predictions")
    axes[0].set_xlabel("Recall")
    axes[0].set_ylabel("Precision")
    axes[0].set_xlim(0, 1)
    axes[0].set_ylim(0, 1)
    axes[0].legend()

    axes[1].plot([0, 1], [0, 1], linestyle="--", linewidth=1, color="black")
    axes[1].set_title("ROC on LOGO predictions")
    axes[1].set_xlabel("False positive rate")
    axes[1].set_ylabel("True positive rate")
    axes[1].set_xlim(0, 1)
    axes[1].set_ylim(0, 1)
    axes[1].legend()

    axes[2].set_title("Hit recovery by ranked screening fraction")
    axes[2].set_xlabel("Top ranked pairs screened (%)")
    axes[2].set_ylabel("Known positives recovered (%)")
    axes[2].set_xlim(0, 100)
    axes[2].set_ylim(0, 100)
    axes[2].legend()

    fig.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved: {output_path}")


def ranking_enrichment_summary(comparison):
    y_true = comparison["CLASS"].astype(int).to_numpy()
    rows = []
    for model_col, label in [
        ("OLD_BASELINE_LOGO_PROB", "old_baseline_logo"),
        ("NEW_IMPROVED_LOGO_PROB", "new_improved_logo"),
    ]:
        probs = comparison[model_col].astype(float).to_numpy()
        screened_fraction, recall_at_rank, precision_at_rank, lift_at_rank = enrichment_curve(
            y_true, probs
        )
        for fraction in [0.01, 0.02, 0.05, 0.10, 0.20]:
            idx = int(np.searchsorted(screened_fraction, fraction, side="left"))
            idx = min(idx, len(screened_fraction) - 1)
            rows.append(
                {
                    "model": label,
                    "top_screened_fraction": fraction,
                    "top_screened_percent": fraction * 100,
                    "precision_at_fraction": float(precision_at_rank[idx]),
                    "known_positive_recall_at_fraction": float(recall_at_rank[idx]),
                    "lift_over_random": float(lift_at_rank[idx]),
                }
            )
    return pd.DataFrame(rows)


def logo_summary(comparison):
    rows = []
    for model_col, label in [
        ("OLD_BASELINE_LOGO_PROB", "old_baseline_logo"),
        ("NEW_IMPROVED_LOGO_PROB", "new_improved_logo"),
    ]:
        for split_name, mask in [
            ("all", np.ones(len(comparison), dtype=bool)),
            ("actual_negative", comparison["CLASS"].astype(int).to_numpy() == 0),
            ("actual_positive", comparison["CLASS"].astype(int).to_numpy() == 1),
        ]:
            arr = comparison.loc[mask, model_col].astype(float).to_numpy()
            rows.append(
                {
                    "model": label,
                    "split": split_name,
                    "count": int(len(arr)),
                    "mean": float(np.mean(arr)),
                    "median": float(np.median(arr)),
                    "p90": float(np.quantile(arr, 0.90)),
                    "p95": float(np.quantile(arr, 0.95)),
                    "p99": float(np.quantile(arr, 0.99)),
                    "max": float(np.max(arr)),
                }
            )
    return pd.DataFrame(rows)


def read_inference_scores(path):
    return pd.read_csv(
        path,
        usecols=["DRUG", "EXCIPIENT", "SOURCE", "PROBABILITY"],
        dtype={"DRUG": str, "EXCIPIENT": str, "SOURCE": str, "PROBABILITY": float},
    )


def plot_inference_distribution(old_scores, new_scores, output_path):
    bins = np.linspace(0, 1, 51)
    fig, ax = plt.subplots(figsize=(9, 6), constrained_layout=True)
    ax.hist(
        old_scores["OLD_INFERENCE_PROB"],
        bins=bins,
        alpha=0.6,
        label=f"Old inference (n={len(old_scores):,})",
        edgecolor="black",
    )
    ax.hist(
        new_scores["NEW_INFERENCE_PROB"],
        bins=bins,
        alpha=0.6,
        label=f"New inference (n={len(new_scores):,})",
        edgecolor="black",
    )
    ax.axvline(0.2, linestyle="--", linewidth=1, label="old threshold 0.20")
    ax.axvline(0.3, linestyle="--", linewidth=1, label="new threshold 0.30")
    ax.set_title("Old vs new inference probability distribution")
    ax.set_xlabel("Predicted probability")
    ax.set_ylabel("Pair count")
    ax.set_xlim(0, 1)
    ax.legend()
    fig.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved: {output_path}")


def plot_inference_scatter(merged, sample_size, output_path):
    sample = merged
    if len(sample) > sample_size:
        sample = sample.sample(n=sample_size, random_state=RANDOM_STATE)

    fig, ax = plt.subplots(figsize=(7, 7), constrained_layout=True)
    ax.scatter(
        sample["OLD_INFERENCE_PROB"],
        sample["NEW_INFERENCE_PROB"],
        s=4,
        alpha=0.25,
    )
    ax.plot([0, 1], [0, 1], color="black", linewidth=1)
    ax.axvline(0.2, linestyle="--", linewidth=1)
    ax.axhline(0.3, linestyle="--", linewidth=1)
    ax.set_title("Old vs new inference probabilities")
    ax.set_xlabel("Old baseline inference probability")
    ax.set_ylabel("New improved inference probability")
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    fig.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved: {output_path}")


def plot_screening_pair_inference_heatmap(
    screen_inference,
    drug_order,
    excipient_order,
    output_path,
):
    if screen_inference["OLD_INFERENCE_PROB"].notna().sum() == 0:
        print("No actual HTS pairs were found in the inference files; skipping HTS-inference heatmap.")
        return

    actual = pivot_matrix(screen_inference, "CLASS", drug_order, excipient_order)
    old_inf = pivot_matrix(screen_inference, "OLD_INFERENCE_PROB", drug_order, excipient_order)
    new_inf = pivot_matrix(screen_inference, "NEW_INFERENCE_PROB", drug_order, excipient_order)
    delta = new_inf - old_inf

    n_drugs = len(drug_order)
    n_excipients = len(excipient_order)
    fig_w = max(12, n_drugs * 0.42)
    fig_h = max(9, n_excipients * 0.24)
    fig, axes = plt.subplots(2, 2, figsize=(fig_w, fig_h), constrained_layout=True)
    axes = axes.ravel()

    plots = [
        (actual, "Actual high-throughput screening", "Greys", 0, 1, "Class"),
        (old_inf, "Old baseline inference probability", "viridis", 0, 1, "Probability"),
        (new_inf, "New improved inference probability", "viridis", 0, 1, "Probability"),
        (delta, "New - old inference probability", "coolwarm", -0.5, 0.5, "Delta"),
    ]

    for ax, (mat, title, cmap, vmin, vmax, cbar_label) in zip(axes, plots):
        im = ax.imshow(
            mat.T.values,
            cmap=cmap,
            vmin=vmin,
            vmax=vmax,
            aspect="auto",
            interpolation="nearest",
        )
        ax.set_title(title)
        ax.set_xlabel("Drugs")
        ax.set_ylabel("Excipients")
        ax.set_xticks(np.arange(n_drugs))
        ax.set_xticklabels(drug_order, rotation=90, fontsize=7)
        ax.set_yticks(np.arange(n_excipients))
        ax.set_yticklabels(excipient_order, fontsize=7)
        cbar = fig.colorbar(im, ax=ax, fraction=0.03, pad=0.02)
        cbar.set_label(cbar_label)

    fig.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved: {output_path}")


def summarize_inference(old_scores, new_scores, merged):
    rows = [
        {
            "comparison": "old_inference",
            "count": int(len(old_scores)),
            "mean": float(old_scores["OLD_INFERENCE_PROB"].mean()),
            "median": float(old_scores["OLD_INFERENCE_PROB"].median()),
            "p95": float(old_scores["OLD_INFERENCE_PROB"].quantile(0.95)),
            "p99": float(old_scores["OLD_INFERENCE_PROB"].quantile(0.99)),
            "max": float(old_scores["OLD_INFERENCE_PROB"].max()),
            "hits_at_native_threshold": int((old_scores["OLD_INFERENCE_PROB"] >= 0.20).sum()),
        },
        {
            "comparison": "new_inference",
            "count": int(len(new_scores)),
            "mean": float(new_scores["NEW_INFERENCE_PROB"].mean()),
            "median": float(new_scores["NEW_INFERENCE_PROB"].median()),
            "p95": float(new_scores["NEW_INFERENCE_PROB"].quantile(0.95)),
            "p99": float(new_scores["NEW_INFERENCE_PROB"].quantile(0.99)),
            "max": float(new_scores["NEW_INFERENCE_PROB"].max()),
            "hits_at_native_threshold": int((new_scores["NEW_INFERENCE_PROB"] >= 0.30).sum()),
        },
        {
            "comparison": "overlap_old_and_new",
            "count": int(len(merged)),
            "mean": float(merged["PROB_DELTA_NEW_MINUS_OLD"].mean()),
            "median": float(merged["PROB_DELTA_NEW_MINUS_OLD"].median()),
            "p95": float(merged["PROB_DELTA_NEW_MINUS_OLD"].quantile(0.95)),
            "p99": float(merged["PROB_DELTA_NEW_MINUS_OLD"].quantile(0.99)),
            "max": float(merged["PROB_DELTA_NEW_MINUS_OLD"].max()),
            "hits_at_native_threshold": int(
                (
                    (merged["OLD_INFERENCE_PROB"] >= 0.20)
                    & (merged["NEW_INFERENCE_PROB"] >= 0.30)
                ).sum()
            ),
        },
    ]
    return pd.DataFrame(rows)


def main():
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    print("Loading high-throughput screening data...")
    screening_data, drugs_smiles, excipients_smiles = load_data(
        SCREENING_FILE, DRUGS_FILE, EXCIPIENTS_FILE
    )
    drug_order, excipient_order = ordered_labels(
        screening_data, drugs_smiles, excipients_smiles
    )

    if not args.skip_logo:
        print("\nBuilding old baseline features: Morgan + RDKit.")
        old_dataset = build_features(
            drugs_smiles,
            excipients_smiles,
            screening_data,
            components=("morgan", "rdkit"),
        )
        old_logo = make_logo_predictions(
            old_dataset,
            components=("morgan", "rdkit"),
            model_name="OLD_BASELINE",
            class_weight=None,
        )

        print("\nBuilding new improved features: Morgan only.")
        new_dataset = build_features(
            drugs_smiles,
            excipients_smiles,
            screening_data,
            components=("morgan",),
        )
        new_logo = make_logo_predictions(
            new_dataset,
            components=("morgan",),
            model_name="NEW_IMPROVED",
            class_weight="balanced",
        )

        logo_comparison = old_logo.merge(
            new_logo[["DRUG", "EXCIPIENT", "NEW_IMPROVED_LOGO_PROB"]],
            on=["DRUG", "EXCIPIENT"],
            how="inner",
        )
        logo_comparison["LOGO_PROB_DELTA_NEW_MINUS_OLD"] = (
            logo_comparison["NEW_IMPROVED_LOGO_PROB"]
            - logo_comparison["OLD_BASELINE_LOGO_PROB"]
        )

        logo_csv = args.output_dir / "logo_actual_old_new_comparison.csv"
        logo_comparison.to_csv(logo_csv, index=False)
        print(f"Saved: {logo_csv}")

        logo_summary_csv = args.output_dir / "logo_probability_summary.csv"
        logo_summary(logo_comparison).to_csv(logo_summary_csv, index=False)
        print(f"Saved: {logo_summary_csv}")

        enrichment_csv = args.output_dir / "logo_ranking_enrichment_summary.csv"
        ranking_enrichment_summary(logo_comparison).to_csv(enrichment_csv, index=False)
        print(f"Saved: {enrichment_csv}")

        plot_combined_old_style_heatmap(
            logo_comparison,
            output_path=args.output_dir / "heatmap_logo_actual_old_new_style.png",
            drug_order=drug_order,
            excipient_order=excipient_order,
        )
        plot_logo_probability_distribution(
            logo_comparison,
            args.output_dir / "prob_dist_logo_old_vs_new.png",
        )
        plot_logo_validation_curves(
            logo_comparison,
            args.output_dir / "logo_validation_curves_old_vs_new.png",
        )

    if args.include_inference_diagnostics:
        print("\nLoading old and new inference scores for unlabeled diagnostics...")
        old_scores = read_inference_scores(args.old_inference_file).rename(
            columns={"PROBABILITY": "OLD_INFERENCE_PROB"}
        )
        new_scores = read_inference_scores(args.new_inference_file).rename(
            columns={"PROBABILITY": "NEW_INFERENCE_PROB"}
        )

        merged = old_scores.merge(
            new_scores,
            on=["DRUG", "EXCIPIENT", "SOURCE"],
            how="inner",
        )
        merged["PROB_DELTA_NEW_MINUS_OLD"] = (
            merged["NEW_INFERENCE_PROB"] - merged["OLD_INFERENCE_PROB"]
        )

        summary_csv = args.output_dir / "inference_probability_summary.csv"
        summarize_inference(old_scores, new_scores, merged).to_csv(summary_csv, index=False)
        print(f"Saved: {summary_csv}")

        top_delta_csv = args.output_dir / "top_inference_probability_changes.csv"
        merged.reindex(
            merged["PROB_DELTA_NEW_MINUS_OLD"].abs().sort_values(ascending=False).index
        ).head(500).to_csv(top_delta_csv, index=False)
        print(f"Saved: {top_delta_csv}")

        screen_pair_scores = (
            merged.groupby(["DRUG", "EXCIPIENT"], as_index=False)[
                ["OLD_INFERENCE_PROB", "NEW_INFERENCE_PROB"]
            ]
            .max()
        )

        screen_inference = screening_data[["DRUG", "EXCIPIENT", "CLASS"]].merge(
            screen_pair_scores,
            on=["DRUG", "EXCIPIENT"],
            how="left",
        )
        screen_inference_csv = args.output_dir / "actual_screening_pairs_inference_scores.csv"
        screen_inference.to_csv(screen_inference_csv, index=False)
        print(f"Saved: {screen_inference_csv}")

        matched = int(screen_inference["OLD_INFERENCE_PROB"].notna().sum())
        print(f"Actual HTS pairs found in inference files: {matched:,}/{len(screen_inference):,}")

        print(
            "Inference diagnostics are unlabeled. Use LOGO plots/curves for model "
            "quality; use these CSV files only to inspect candidate-list changes."
        )

    print(f"\nDone. Final comparison figures saved in: {args.output_dir}")


if __name__ == "__main__":
    main()
