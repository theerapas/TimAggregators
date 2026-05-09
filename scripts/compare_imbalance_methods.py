import argparse
import pathlib
import sys
import warnings
from dataclasses import dataclass

PROJECT_ROOT = pathlib.Path(__file__).resolve().parent.parent
sys.path.append(str(PROJECT_ROOT))

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import LeaveOneGroupOut, StratifiedKFold

from src.timaggregators.evaluation import compute_metrics, get_probabilities
from src.timaggregators.features import build_features
from src.timaggregators.loaders import load_data

try:
    from imblearn.over_sampling import RandomOverSampler, SMOTE
except Exception:
    RandomOverSampler = None
    SMOTE = None

warnings.filterwarnings("ignore")


DATA_DIR = PROJECT_ROOT / "data" / "raw"
SCREENING_FILE = DATA_DIR / "screening_data.tsv"
DRUGS_FILE = DATA_DIR / "selected_drugs_smiles.tsv"
EXCIPIENTS_FILE = DATA_DIR / "selected_excipients_smiles.tsv"
OUT_DIR = PROJECT_ROOT / "results" / "imbalance"

RANDOM_STATE = 42
N_ESTIMATORS = 500
DEFAULT_THRESHOLDS = [0.2, 0.5]
THRESHOLD_SWEEP = [round(x, 2) for x in np.arange(0.05, 0.95, 0.05)]
METRIC_COLS = ["MCC", "F1", "Precision", "Accuracy", "Recall", "AUROC", "AUPRC"]


@dataclass(frozen=True)
class ImbalanceMethod:
    name: str
    stage: str
    class_weight: str | None = None
    sampler: str | None = None
    threshold_mode: str = "fixed"


METHODS = {
    "A0_no_imbalance": ImbalanceMethod(
        name="A0_no_imbalance",
        stage="A0",
    ),
    "B1_class_weight": ImbalanceMethod(
        name="B1_class_weight",
        stage="B1",
        class_weight="balanced",
    ),
    "B2_random_oversampling": ImbalanceMethod(
        name="B2_random_oversampling",
        stage="B2",
        sampler="random_oversampling",
    ),
    "B3_smote": ImbalanceMethod(
        name="B3_smote",
        stage="B3",
        sampler="smote",
    ),
    "B4_threshold_tuning_only": ImbalanceMethod(
        name="B4_threshold_tuning_only",
        stage="B4",
        threshold_mode="sweep",
    ),
    "B5_class_weight_threshold_tuning": ImbalanceMethod(
        name="B5_class_weight_threshold_tuning",
        stage="B5",
        class_weight="balanced",
        threshold_mode="sweep",
    ),
}


def parse_args():
    parser = argparse.ArgumentParser(
        description="Stage B ablation: compare imbalance handling strategies."
    )
    parser.add_argument(
        "--methods",
        nargs="+",
        default=list(METHODS),
        choices=list(METHODS),
        help="Subset of imbalance methods to run.",
    )
    parser.add_argument(
        "--n-splits",
        type=int,
        default=10,
        help="Number of stratified CV folds.",
    )
    parser.add_argument(
        "--skip-logo",
        action="store_true",
        help="Only run stratified CV. Useful for quick smoke tests.",
    )
    parser.add_argument(
        "--max-logo-folds",
        type=int,
        default=None,
        help="Limit LOGO folds for debugging. Omit for the full LOGO study.",
    )
    return parser.parse_args()


def make_random_forest(class_weight: str | None = None):
    return RandomForestClassifier(
        n_estimators=N_ESTIMATORS,
        random_state=RANDOM_STATE,
        n_jobs=-1,
        class_weight=class_weight,
    )


def make_sampler(method: ImbalanceMethod, y_train: pd.Series):
    if method.sampler is None:
        return None

    if method.sampler == "random_oversampling":
        require_imblearn("RandomOverSampler", RandomOverSampler)
        return RandomOverSampler(random_state=RANDOM_STATE)

    if method.sampler == "smote":
        require_imblearn("SMOTE", SMOTE)
        positive_count = int(np.sum(np.asarray(y_train) == 1))
        if positive_count < 2:
            return None
        k_neighbors = min(5, positive_count - 1)
        return SMOTE(random_state=RANDOM_STATE, k_neighbors=k_neighbors)

    raise ValueError(f"Unknown sampler: {method.sampler}")


def require_imblearn(name: str, obj):
    if obj is None:
        raise ImportError(
            f"{name} requires imbalanced-learn. Install it with the updated "
            "environment.yml before running this method."
        )


def thresholds_for_method(method: ImbalanceMethod):
    if method.threshold_mode == "sweep":
        return THRESHOLD_SWEEP
    return DEFAULT_THRESHOLDS


def fit_and_score_fold(
    X_train: pd.DataFrame,
    y_train: pd.Series,
    X_test: pd.DataFrame,
    method: ImbalanceMethod,
):
    sampler = make_sampler(method, y_train)
    if sampler is not None:
        X_fit, y_fit = sampler.fit_resample(X_train, y_train)
    else:
        X_fit, y_fit = X_train, y_train

    model = make_random_forest(class_weight=method.class_weight)
    model.fit(X_fit, y_fit)
    return get_probabilities(model, X_test)


def evaluate_cv(
    X: pd.DataFrame,
    y: pd.Series,
    method: ImbalanceMethod,
    n_splits: int,
):
    rows = []
    skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=RANDOM_STATE)
    thresholds = thresholds_for_method(method)

    for fold, (train_idx, test_idx) in enumerate(skf.split(X, y), start=1):
        print(f"  [CV] {method.name} fold {fold}/{n_splits}")
        X_train, X_test = X.iloc[train_idx], X.iloc[test_idx]
        y_train, y_test = y.iloc[train_idx], y.iloc[test_idx]

        y_prob = fit_and_score_fold(X_train, y_train, X_test, method)
        rows.extend(
            metrics_rows(
                y_true=y_test,
                y_prob=y_prob,
                thresholds=thresholds,
                method=method,
                evaluation="CV",
                fold=fold,
            )
        )

    return pd.DataFrame(rows)


def evaluate_logo(
    X: pd.DataFrame,
    y: pd.Series,
    groups: pd.Series,
    method: ImbalanceMethod,
    max_folds: int | None = None,
):
    rows = []
    logo = LeaveOneGroupOut()
    total_folds = groups.nunique()
    thresholds = thresholds_for_method(method)

    for fold, (train_idx, test_idx) in enumerate(
        logo.split(X, y, groups=groups), start=1
    ):
        if max_folds is not None and fold > max_folds:
            break

        held_out_drug = groups.iloc[test_idx].iloc[0]
        print(f"  [LOGO] {method.name} fold {fold}/{total_folds} - held out: {held_out_drug}")

        X_train, X_test = X.iloc[train_idx], X.iloc[test_idx]
        y_train, y_test = y.iloc[train_idx], y.iloc[test_idx]

        y_prob = fit_and_score_fold(X_train, y_train, X_test, method)
        rows.extend(
            metrics_rows(
                y_true=y_test,
                y_prob=y_prob,
                thresholds=thresholds,
                method=method,
                evaluation="LOGO",
                fold=fold,
                held_out_drug=held_out_drug,
            )
        )

    return pd.DataFrame(rows)


def metrics_rows(
    y_true,
    y_prob,
    thresholds: list[float],
    method: ImbalanceMethod,
    evaluation: str,
    fold: int,
    held_out_drug: str | None = None,
):
    rows = []
    for threshold in thresholds:
        y_pred = (y_prob >= threshold).astype(int)
        metrics = compute_metrics(y_true, y_pred, y_prob)
        metrics.update(
            {
                "Stage": method.stage,
                "Method": method.name,
                "Evaluation": evaluation,
                "Fold": fold,
                "Threshold": threshold,
                "ClassWeight": method.class_weight or "none",
                "Sampler": method.sampler or "none",
                "ThresholdMode": method.threshold_mode,
            }
        )
        if held_out_drug is not None:
            metrics["HeldOutDrug"] = held_out_drug
        rows.append(metrics)
    return rows


def summarize_results(df: pd.DataFrame):
    summary = (
        df.groupby(["Evaluation", "Stage", "Method", "Threshold"])[METRIC_COLS]
        .agg(["mean", "std"])
        .round(4)
    )
    summary.columns = [f"{metric}_{stat}" for metric, stat in summary.columns]
    return summary.reset_index()


def make_leaderboard(summary: pd.DataFrame):
    return summary.sort_values(
        by=[
            "Evaluation",
            "AUPRC_mean",
            "MCC_mean",
            "F1_mean",
            "Recall_mean",
            "Precision_mean",
        ],
        ascending=[True, False, False, False, False, False],
    ).reset_index(drop=True)


def load_stage_a_dataset():
    screening_data, drugs_smiles, excipients_smiles = load_data(
        SCREENING_FILE, DRUGS_FILE, EXCIPIENTS_FILE
    )
    dataset = build_features(drugs_smiles, excipients_smiles, screening_data)

    X = dataset.drop(columns=["DRUG", "EXCIPIENT", "CLASS"]).copy()
    X = X.replace([np.inf, -np.inf], np.nan).fillna(0.0).astype(np.float32)
    y = dataset["CLASS"].astype(int).copy()
    groups = dataset["DRUG"].copy()

    return X, y, groups


def main():
    args = parse_args()
    selected_methods = [METHODS[name] for name in args.methods]

    print("Loading Stage A feature set...")
    X, y, groups = load_stage_a_dataset()
    print(f"Final dataset shape: {X.shape}")
    print(f"Class counts: {dict(y.value_counts().sort_index())}")

    all_results = []
    for method in selected_methods:
        print(f"\nRunning {method.stage}: {method.name}")
        cv_df = evaluate_cv(X, y, method, n_splits=args.n_splits)
        all_results.append(cv_df)

        if not args.skip_logo:
            logo_df = evaluate_logo(
                X,
                y,
                groups,
                method,
                max_folds=args.max_logo_folds,
            )
            all_results.append(logo_df)

    results = pd.concat(all_results, ignore_index=True)
    summary = summarize_results(results)
    leaderboard = make_leaderboard(summary)

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    results.to_csv(OUT_DIR / "imbalance_all_folds.csv", index=False)
    summary.to_csv(OUT_DIR / "imbalance_summary.csv", index=False)
    leaderboard.to_csv(OUT_DIR / "imbalance_leaderboard.csv", index=False)

    print("\nSaved:")
    print(f"- {OUT_DIR / 'imbalance_all_folds.csv'}")
    print(f"- {OUT_DIR / 'imbalance_summary.csv'}")
    print(f"- {OUT_DIR / 'imbalance_leaderboard.csv'}")
    print("\nTop leaderboard rows:")
    print(leaderboard.head(12).to_string(index=False))


if __name__ == "__main__":
    main()
