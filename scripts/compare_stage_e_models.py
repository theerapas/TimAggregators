import argparse
import pathlib
import sys
import warnings
from dataclasses import dataclass

PROJECT_ROOT = pathlib.Path(__file__).resolve().parent.parent
sys.path.append(str(PROJECT_ROOT))

import numpy as np
import pandas as pd
from sklearn.ensemble import (
    ExtraTreesClassifier,
    HistGradientBoostingClassifier,
    RandomForestClassifier,
)
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import LeaveOneGroupOut, StratifiedKFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.utils.class_weight import compute_sample_weight

from scripts.compare_representations import (
    REPRESENTATIONS,
    build_representation_dataset,
    dataset_to_xy,
    read_embedding_file,
)
from src.timaggregators.evaluation import compute_metrics, get_probabilities
from src.timaggregators.loaders import load_data

try:
    from xgboost import XGBClassifier
except Exception:
    XGBClassifier = None

try:
    from lightgbm import LGBMClassifier
except Exception:
    LGBMClassifier = None

warnings.filterwarnings("ignore")


DATA_DIR = PROJECT_ROOT / "data" / "raw"
SCREENING_FILE = DATA_DIR / "screening_data.tsv"
DRUGS_FILE = DATA_DIR / "selected_drugs_smiles.tsv"
EXCIPIENTS_FILE = DATA_DIR / "selected_excipients_smiles.tsv"
OUT_DIR = PROJECT_ROOT / "results" / "stage_e_models"

RANDOM_STATE = 42
THRESHOLD_SWEEP = [round(x, 2) for x in np.arange(0.05, 0.95, 0.05)]
METRIC_COLS = ["MCC", "F1", "Precision", "Accuracy", "Recall", "AUROC", "AUPRC"]


@dataclass(frozen=True)
class FeatureSetup:
    name: str
    representation_key: str
    purpose: str
    imbalance_stage: str
    use_class_balance: bool


FEATURE_SETUPS = {
    "ranking_c1_b5": FeatureSetup(
        name="ranking_c1_b5",
        representation_key="C1_morgan_only",
        purpose="ranking",
        imbalance_stage="B5",
        use_class_balance=True,
    ),
    "decision_c7_b4": FeatureSetup(
        name="decision_c7_b4",
        representation_key="C7_morgan_chemberta",
        purpose="thresholded_decision",
        imbalance_stage="B4",
        use_class_balance=False,
    ),
}

DEFAULT_MODELS = [
    "RandomForest",
    "ExtraTrees",
    "LogisticRegression",
    "KernelSVM",
    "HistGradientBoosting",
    "XGBoost",
    "LightGBM",
]


def parse_args():
    parser = argparse.ArgumentParser(
        description="Stage E ablation: compare models on selected Stage C feature setups."
    )
    parser.add_argument(
        "--feature-setups",
        nargs="+",
        default=list(FEATURE_SETUPS),
        choices=list(FEATURE_SETUPS),
        help="Stage C carry-forward feature setups to evaluate.",
    )
    parser.add_argument(
        "--models",
        nargs="+",
        default=DEFAULT_MODELS,
        choices=DEFAULT_MODELS,
        help="Models to compare.",
    )
    parser.add_argument(
        "--chemberta-file",
        type=pathlib.Path,
        default=PROJECT_ROOT / "data" / "processed" / "chemberta_embeddings.csv",
        help="ChemBERTa embedding CSV for C7.",
    )
    parser.add_argument(
        "--embedding-key-col",
        default="NAME",
        help="Column used to join embedding files to selected molecules.",
    )
    parser.add_argument("--n-splits", type=int, default=10)
    parser.add_argument("--skip-logo", action="store_true")
    parser.add_argument("--max-logo-folds", type=int, default=None)
    return parser.parse_args()


def scale_pos_weight(y_train):
    pos = int(np.sum(np.asarray(y_train) == 1))
    neg = int(np.sum(np.asarray(y_train) == 0))
    return max(1.0, neg / max(pos, 1))


def make_model(model_name: str, y_train: pd.Series, use_class_balance: bool):
    class_weight = "balanced" if use_class_balance else None

    if model_name == "RandomForest":
        return RandomForestClassifier(
            n_estimators=500,
            random_state=RANDOM_STATE,
            n_jobs=-1,
            class_weight=class_weight,
        )

    if model_name == "ExtraTrees":
        return ExtraTreesClassifier(
            n_estimators=500,
            random_state=RANDOM_STATE,
            n_jobs=-1,
            class_weight=class_weight,
        )

    if model_name == "LogisticRegression":
        return Pipeline(
            [
                ("scaler", StandardScaler()),
                (
                    "model",
                    LogisticRegression(
                        max_iter=3000,
                        solver="liblinear",
                        random_state=RANDOM_STATE,
                        class_weight=class_weight,
                    ),
                ),
            ]
        )

    if model_name == "KernelSVM":
        return Pipeline(
            [
                ("scaler", StandardScaler()),
                (
                    "model",
                    SVC(
                        kernel="rbf",
                        C=1.0,
                        gamma="scale",
                        probability=True,
                        random_state=RANDOM_STATE,
                        class_weight=class_weight,
                    ),
                ),
            ]
        )

    if model_name == "HistGradientBoosting":
        return HistGradientBoostingClassifier(
            max_iter=300,
            learning_rate=0.05,
            max_depth=8,
            random_state=RANDOM_STATE,
        )

    if model_name == "XGBoost":
        if XGBClassifier is None:
            raise ImportError("XGBoost is not installed.")
        return XGBClassifier(
            n_estimators=500,
            learning_rate=0.05,
            max_depth=6,
            min_child_weight=1,
            subsample=0.8,
            colsample_bytree=0.8,
            reg_lambda=1.0,
            objective="binary:logistic",
            eval_metric="logloss",
            random_state=RANDOM_STATE,
            n_jobs=-1,
            tree_method="hist",
            scale_pos_weight=scale_pos_weight(y_train) if use_class_balance else 1.0,
        )

    if model_name == "LightGBM":
        if LGBMClassifier is None:
            raise ImportError("LightGBM is not installed.")
        return LGBMClassifier(
            n_estimators=500,
            learning_rate=0.05,
            num_leaves=31,
            subsample=0.8,
            colsample_bytree=0.8,
            objective="binary",
            class_weight=class_weight,
            random_state=RANDOM_STATE,
            n_jobs=-1,
            verbose=-1,
        )

    raise ValueError(f"Unknown model: {model_name}")


def fit_model(model, model_name: str, X_train, y_train, use_class_balance: bool):
    if model_name == "HistGradientBoosting" and use_class_balance:
        weights = compute_sample_weight(class_weight="balanced", y=y_train)
        model.fit(X_train, y_train, sample_weight=weights)
    else:
        model.fit(X_train, y_train)
    return model


def metrics_rows(
    y_true,
    y_prob,
    feature_setup: FeatureSetup,
    model_name: str,
    evaluation: str,
    fold: int,
    held_out_drug: str | None = None,
):
    rows = []
    for threshold in THRESHOLD_SWEEP:
        y_pred = (y_prob >= threshold).astype(int)
        metrics = compute_metrics(y_true, y_pred, y_prob)
        metrics.update(
            {
                "FeatureSetup": feature_setup.name,
                "Purpose": feature_setup.purpose,
                "RepresentationStage": REPRESENTATIONS[
                    feature_setup.representation_key
                ].stage,
                "Representation": feature_setup.representation_key,
                "ImbalanceStage": feature_setup.imbalance_stage,
                "ClassBalance": feature_setup.use_class_balance,
                "Model": model_name,
                "Evaluation": evaluation,
                "Fold": fold,
                "Threshold": threshold,
            }
        )
        if held_out_drug is not None:
            metrics["HeldOutDrug"] = held_out_drug
        rows.append(metrics)
    return rows


def evaluate_cv(
    X: pd.DataFrame,
    y: pd.Series,
    feature_setup: FeatureSetup,
    model_name: str,
    n_splits: int,
):
    rows = []
    skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=RANDOM_STATE)

    for fold, (train_idx, test_idx) in enumerate(skf.split(X, y), start=1):
        print(f"  [CV] {feature_setup.name}/{model_name} fold {fold}/{n_splits}")
        X_train, X_test = X.iloc[train_idx], X.iloc[test_idx]
        y_train, y_test = y.iloc[train_idx], y.iloc[test_idx]

        model = make_model(model_name, y_train, feature_setup.use_class_balance)
        model = fit_model(
            model,
            model_name,
            X_train,
            y_train,
            feature_setup.use_class_balance,
        )
        y_prob = get_probabilities(model, X_test)
        rows.extend(metrics_rows(y_test, y_prob, feature_setup, model_name, "CV", fold))

    return pd.DataFrame(rows)


def evaluate_logo(
    X: pd.DataFrame,
    y: pd.Series,
    groups: pd.Series,
    feature_setup: FeatureSetup,
    model_name: str,
    max_folds: int | None = None,
):
    rows = []
    logo = LeaveOneGroupOut()
    total_folds = groups.nunique()

    for fold, (train_idx, test_idx) in enumerate(
        logo.split(X, y, groups=groups), start=1
    ):
        if max_folds is not None and fold > max_folds:
            break

        held_out_drug = groups.iloc[test_idx].iloc[0]
        print(
            f"  [LOGO] {feature_setup.name}/{model_name} "
            f"fold {fold}/{total_folds} - held out: {held_out_drug}"
        )
        X_train, X_test = X.iloc[train_idx], X.iloc[test_idx]
        y_train, y_test = y.iloc[train_idx], y.iloc[test_idx]

        model = make_model(model_name, y_train, feature_setup.use_class_balance)
        model = fit_model(
            model,
            model_name,
            X_train,
            y_train,
            feature_setup.use_class_balance,
        )
        y_prob = get_probabilities(model, X_test)
        rows.extend(
            metrics_rows(
                y_test,
                y_prob,
                feature_setup,
                model_name,
                "LOGO",
                fold,
                held_out_drug,
            )
        )

    return pd.DataFrame(rows)


def summarize_results(df: pd.DataFrame):
    summary = (
        df.groupby(
            [
                "Evaluation",
                "FeatureSetup",
                "Purpose",
                "RepresentationStage",
                "Representation",
                "ImbalanceStage",
                "ClassBalance",
                "Model",
                "Threshold",
            ]
        )[METRIC_COLS]
        .agg(["mean", "std"])
        .round(4)
    )
    summary.columns = [f"{metric}_{stat}" for metric, stat in summary.columns]
    return summary.reset_index()


def make_leaderboard(summary: pd.DataFrame):
    return summary.sort_values(
        by=[
            "Evaluation",
            "Purpose",
            "AUPRC_mean",
            "MCC_mean",
            "F1_mean",
            "Recall_mean",
            "Precision_mean",
        ],
        ascending=[True, True, False, False, False, False, False],
    ).reset_index(drop=True)


def load_embedding_tables(args):
    tables = {}
    if any(
        FEATURE_SETUPS[name].representation_key == "C7_morgan_chemberta"
        for name in args.feature_setups
    ):
        if not args.chemberta_file.exists():
            raise FileNotFoundError(
                f"ChemBERTa file not found: {args.chemberta_file}. "
                "Run scripts/generate_chemberta_embeddings.py first."
            )
        tables["chemberta"] = read_embedding_file(
            args.chemberta_file, args.embedding_key_col
        )
    return tables


def main():
    args = parse_args()
    selected_setups = [FEATURE_SETUPS[name] for name in args.feature_setups]
    embedding_tables = load_embedding_tables(args)

    print("Loading data...")
    screening_data, drugs_smiles, excipients_smiles = load_data(
        SCREENING_FILE, DRUGS_FILE, EXCIPIENTS_FILE
    )

    all_results = []
    for feature_setup in selected_setups:
        representation = REPRESENTATIONS[feature_setup.representation_key]
        print(f"\nBuilding feature setup: {feature_setup.name}")
        dataset = build_representation_dataset(
            screening_data,
            drugs_smiles,
            excipients_smiles,
            representation,
            embedding_tables,
            key_col=args.embedding_key_col,
        )
        X, y, groups = dataset_to_xy(dataset)
        print(f"Final dataset shape: {X.shape}")
        print(f"Class counts: {dict(y.value_counts().sort_index())}")

        for model_name in args.models:
            try:
                print(f"\nRunning {feature_setup.name} with {model_name}")
                cv_df = evaluate_cv(X, y, feature_setup, model_name, args.n_splits)
                all_results.append(cv_df)

                if not args.skip_logo:
                    logo_df = evaluate_logo(
                        X,
                        y,
                        groups,
                        feature_setup,
                        model_name,
                        max_folds=args.max_logo_folds,
                    )
                    all_results.append(logo_df)
            except ImportError as exc:
                print(f"Skipping {model_name}: {exc}")

    if not all_results:
        raise RuntimeError("No Stage E model results were produced.")

    results = pd.concat(all_results, ignore_index=True)
    summary = summarize_results(results)
    leaderboard = make_leaderboard(summary)

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    results.to_csv(OUT_DIR / "stage_e_model_all_folds.csv", index=False)
    summary.to_csv(OUT_DIR / "stage_e_model_summary.csv", index=False)
    leaderboard.to_csv(OUT_DIR / "stage_e_model_leaderboard.csv", index=False)

    print("\nSaved:")
    print(f"- {OUT_DIR / 'stage_e_model_all_folds.csv'}")
    print(f"- {OUT_DIR / 'stage_e_model_summary.csv'}")
    print(f"- {OUT_DIR / 'stage_e_model_leaderboard.csv'}")
    print("\nTop leaderboard rows:")
    print(leaderboard.head(20).to_string(index=False))


if __name__ == "__main__":
    main()
