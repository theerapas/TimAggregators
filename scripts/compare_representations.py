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

warnings.filterwarnings("ignore")


DATA_DIR = PROJECT_ROOT / "data" / "raw"
SCREENING_FILE = DATA_DIR / "screening_data.tsv"
DRUGS_FILE = DATA_DIR / "selected_drugs_smiles.tsv"
EXCIPIENTS_FILE = DATA_DIR / "selected_excipients_smiles.tsv"
OUT_DIR = PROJECT_ROOT / "results" / "representations"

RANDOM_STATE = 42
N_ESTIMATORS = 500
THRESHOLD_SWEEP = [round(x, 2) for x in np.arange(0.05, 0.95, 0.05)]
METRIC_COLS = ["MCC", "F1", "Precision", "Accuracy", "Recall", "AUROC", "AUPRC"]


@dataclass(frozen=True)
class RepresentationSpec:
    name: str
    stage: str
    components: tuple[str, ...] = ()
    embeddings: tuple[str, ...] = ()


@dataclass(frozen=True)
class StageBMethod:
    name: str
    stage: str
    class_weight: str | None = None


REPRESENTATIONS = {
    "C1_morgan_only": RepresentationSpec(
        name="C1_morgan_only",
        stage="C1",
        components=("morgan",),
    ),
    "C2_rdkit_descriptors_only": RepresentationSpec(
        name="C2_rdkit_descriptors_only",
        stage="C2",
        components=("rdkit",),
    ),
    "C3_morgan_rdkit": RepresentationSpec(
        name="C3_morgan_rdkit",
        stage="C3",
        components=("morgan", "rdkit"),
    ),
    "C4_eos2lm8": RepresentationSpec(
        name="C4_eos2lm8",
        stage="C4",
        embeddings=("eos2lm8",),
    ),
    "C5_chemberta": RepresentationSpec(
        name="C5_chemberta",
        stage="C5",
        embeddings=("chemberta",),
    ),
    "C6_morgan_eos2lm8": RepresentationSpec(
        name="C6_morgan_eos2lm8",
        stage="C6",
        components=("morgan",),
        embeddings=("eos2lm8",),
    ),
    "C7_morgan_chemberta": RepresentationSpec(
        name="C7_morgan_chemberta",
        stage="C7",
        components=("morgan",),
        embeddings=("chemberta",),
    ),
    "C8_morgan_rdkit_eos2lm8": RepresentationSpec(
        name="C8_morgan_rdkit_eos2lm8",
        stage="C8",
        components=("morgan", "rdkit"),
        embeddings=("eos2lm8",),
    ),
    "C9_morgan_rdkit_chemberta": RepresentationSpec(
        name="C9_morgan_rdkit_chemberta",
        stage="C9",
        components=("morgan", "rdkit"),
        embeddings=("chemberta",),
    ),
    "C10_unimap": RepresentationSpec(
        name="C10_unimap",
        stage="C10",
        embeddings=("unimap",),
    ),
    "C11_morgan_unimap": RepresentationSpec(
        name="C11_morgan_unimap",
        stage="C11",
        components=("morgan",),
        embeddings=("unimap",),
    ),
    "C12_morgan_rdkit_unimap": RepresentationSpec(
        name="C12_morgan_rdkit_unimap",
        stage="C12",
        components=("morgan", "rdkit"),
        embeddings=("unimap",),
    ),
}

METHODS = {
    "B4_threshold_tuning_only": StageBMethod(
        name="B4_threshold_tuning_only",
        stage="B4",
    ),
    "B5_class_weight_threshold_tuning": StageBMethod(
        name="B5_class_weight_threshold_tuning",
        stage="B5",
        class_weight="balanced",
    ),
}


def parse_args():
    parser = argparse.ArgumentParser(
        description="Stage C ablation: compare molecular representations."
    )
    parser.add_argument(
        "--representations",
        nargs="+",
        default=list(REPRESENTATIONS),
        choices=list(REPRESENTATIONS),
        help="Subset of Stage C representations to run.",
    )
    parser.add_argument(
        "--methods",
        nargs="+",
        default=list(METHODS),
        choices=list(METHODS),
        help="Subset of selected Stage B methods to run.",
    )
    parser.add_argument(
        "--eos2lm8-file",
        type=pathlib.Path,
        default=None,
        help="Optional CSV/TSV embedding file keyed by molecule name for C4/C6/C8.",
    )
    parser.add_argument(
        "--chemberta-file",
        type=pathlib.Path,
        default=None,
        help="Optional CSV/TSV embedding file keyed by molecule name for C5/C7/C9.",
    )
    parser.add_argument(
        "--unimap-file",
        type=pathlib.Path,
        default=None,
        help="Optional CSV/TSV embedding file keyed by molecule name for C10/C11/C12.",
    )
    parser.add_argument(
        "--embedding-key-col",
        default="NAME",
        help="Column used to join embedding files to selected molecules.",
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


def make_random_forest(method: StageBMethod):
    return RandomForestClassifier(
        n_estimators=N_ESTIMATORS,
        random_state=RANDOM_STATE,
        n_jobs=-1,
        class_weight=method.class_weight,
    )


def read_embedding_file(path: pathlib.Path, key_col: str):
    sep = "\t" if path.suffix.lower() in {".tsv", ".txt"} else ","
    df = pd.read_csv(path, sep=sep)
    if key_col not in df.columns:
        raise ValueError(f"{path} must contain key column {key_col!r}.")

    feature_cols = [
        col
        for col in df.columns
        if col != key_col and pd.api.types.is_numeric_dtype(df[col])
    ]
    if not feature_cols:
        raise ValueError(f"{path} does not contain numeric embedding columns.")

    return df[[key_col] + feature_cols].drop_duplicates(subset=[key_col])


def load_embedding_tables(args):
    tables = {}
    if args.eos2lm8_file is not None:
        tables["eos2lm8"] = read_embedding_file(args.eos2lm8_file, args.embedding_key_col)
    if args.chemberta_file is not None:
        tables["chemberta"] = read_embedding_file(
            args.chemberta_file, args.embedding_key_col
        )
    if args.unimap_file is not None:
        tables["unimap"] = read_embedding_file(args.unimap_file, args.embedding_key_col)
    return tables


def unavailable_reason(representation: RepresentationSpec, embedding_tables: dict):
    missing = [name for name in representation.embeddings if name not in embedding_tables]
    if not missing:
        return None
    return "missing embedding file(s): " + ", ".join(missing)


def embedding_features_for_molecules(
    molecules: pd.DataFrame,
    embedding_table: pd.DataFrame,
    embedding_name: str,
    prefix: str,
    key_col: str,
):
    merged = molecules[["NAME"]].merge(
        embedding_table,
        left_on="NAME",
        right_on=key_col,
        how="left",
    )

    non_feature_cols = ["NAME"]
    if key_col != "NAME":
        non_feature_cols.append(key_col)
    feature_cols = [col for col in merged.columns if col not in non_feature_cols]

    missing = merged.loc[merged[feature_cols].isna().all(axis=1), "NAME"]
    if len(missing) > 0:
        preview = ", ".join(missing.head(10).astype(str))
        raise ValueError(
            f"{embedding_name} embeddings are missing for {len(missing)} molecule(s): "
            f"{preview}"
        )

    feature_df = merged[feature_cols]
    feature_df = feature_df.replace([np.inf, -np.inf], np.nan).fillna(0.0)
    feature_df.columns = [
        f"{prefix}_{embedding_name}_{col}" for col in feature_df.columns
    ]
    return pd.concat([molecules[["NAME"]].reset_index(drop=True), feature_df], axis=1)


def build_representation_dataset(
    screening_data: pd.DataFrame,
    drugs_smiles: pd.DataFrame,
    excipients_smiles: pd.DataFrame,
    representation: RepresentationSpec,
    embedding_tables: dict,
    key_col: str,
):
    feature_blocks = []

    if representation.components:
        feature_blocks.append(
            build_features(
                drugs_smiles,
                excipients_smiles,
                screening_data,
                components=representation.components,
            )
        )
    else:
        feature_blocks.append(screening_data[["DRUG", "EXCIPIENT", "CLASS"]].copy())

    dataset = feature_blocks[0]

    for embedding_name in representation.embeddings:
        embedding_table = embedding_tables[embedding_name]
        drug_embedding_df = embedding_features_for_molecules(
            drugs_smiles,
            embedding_table,
            embedding_name=embedding_name,
            prefix="Drug",
            key_col=key_col,
        )
        exc_embedding_df = embedding_features_for_molecules(
            excipients_smiles,
            embedding_table,
            embedding_name=embedding_name,
            prefix="Exc",
            key_col=key_col,
        )

        dataset = dataset.merge(
            drug_embedding_df,
            left_on="DRUG",
            right_on="NAME",
            how="left",
        ).drop(columns=["NAME"])
        dataset = dataset.merge(
            exc_embedding_df,
            left_on="EXCIPIENT",
            right_on="NAME",
            how="left",
        ).drop(columns=["NAME"])

    return dataset


def dataset_to_xy(dataset: pd.DataFrame):
    X = dataset.drop(columns=["DRUG", "EXCIPIENT", "CLASS"]).copy()
    X = X.replace([np.inf, -np.inf], np.nan).fillna(0.0).astype(np.float32)
    y = dataset["CLASS"].astype(int).copy()
    groups = dataset["DRUG"].copy()
    return X, y, groups


def fit_and_score_fold(
    X_train: pd.DataFrame,
    y_train: pd.Series,
    X_test: pd.DataFrame,
    method: StageBMethod,
):
    model = make_random_forest(method)
    model.fit(X_train, y_train)
    return get_probabilities(model, X_test)


def evaluate_cv(
    X: pd.DataFrame,
    y: pd.Series,
    representation: RepresentationSpec,
    method: StageBMethod,
    n_splits: int,
):
    rows = []
    skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=RANDOM_STATE)

    for fold, (train_idx, test_idx) in enumerate(skf.split(X, y), start=1):
        print(f"  [CV] {representation.stage}/{method.stage} fold {fold}/{n_splits}")
        X_train, X_test = X.iloc[train_idx], X.iloc[test_idx]
        y_train, y_test = y.iloc[train_idx], y.iloc[test_idx]

        y_prob = fit_and_score_fold(X_train, y_train, X_test, method)
        rows.extend(
            metrics_rows(
                y_true=y_test,
                y_prob=y_prob,
                thresholds=THRESHOLD_SWEEP,
                representation=representation,
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
    representation: RepresentationSpec,
    method: StageBMethod,
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
            f"  [LOGO] {representation.stage}/{method.stage} "
            f"fold {fold}/{total_folds} - held out: {held_out_drug}"
        )

        X_train, X_test = X.iloc[train_idx], X.iloc[test_idx]
        y_train, y_test = y.iloc[train_idx], y.iloc[test_idx]

        y_prob = fit_and_score_fold(X_train, y_train, X_test, method)
        rows.extend(
            metrics_rows(
                y_true=y_test,
                y_prob=y_prob,
                thresholds=THRESHOLD_SWEEP,
                representation=representation,
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
    representation: RepresentationSpec,
    method: StageBMethod,
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
                "RepresentationStage": representation.stage,
                "Representation": representation.name,
                "ImbalanceStage": method.stage,
                "ImbalanceMethod": method.name,
                "Evaluation": evaluation,
                "Fold": fold,
                "Threshold": threshold,
                "ClassWeight": method.class_weight or "none",
            }
        )
        if held_out_drug is not None:
            metrics["HeldOutDrug"] = held_out_drug
        rows.append(metrics)
    return rows


def summarize_results(df: pd.DataFrame):
    summary = (
        df.groupby(
            [
                "Evaluation",
                "RepresentationStage",
                "Representation",
                "ImbalanceStage",
                "ImbalanceMethod",
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
            "AUPRC_mean",
            "MCC_mean",
            "F1_mean",
            "Recall_mean",
            "Precision_mean",
        ],
        ascending=[True, False, False, False, False, False],
    ).reset_index(drop=True)


def main():
    args = parse_args()
    embedding_tables = load_embedding_tables(args)
    selected_representations = [REPRESENTATIONS[name] for name in args.representations]
    selected_methods = [METHODS[name] for name in args.methods]

    print("Loading data...")
    screening_data, drugs_smiles, excipients_smiles = load_data(
        SCREENING_FILE, DRUGS_FILE, EXCIPIENTS_FILE
    )

    all_results = []
    skipped = []
    for representation in selected_representations:
        reason = unavailable_reason(representation, embedding_tables)
        if reason is not None:
            print(f"\nSkipping {representation.stage}: {representation.name} ({reason})")
            skipped.append(
                {
                    "RepresentationStage": representation.stage,
                    "Representation": representation.name,
                    "Reason": reason,
                }
            )
            continue

        print(f"\nBuilding {representation.stage}: {representation.name}")
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

        for method in selected_methods:
            print(f"\nRunning {representation.stage} with {method.stage}: {method.name}")
            cv_df = evaluate_cv(
                X,
                y,
                representation,
                method,
                n_splits=args.n_splits,
            )
            all_results.append(cv_df)

            if not args.skip_logo:
                logo_df = evaluate_logo(
                    X,
                    y,
                    groups,
                    representation,
                    method,
                    max_folds=args.max_logo_folds,
                )
                all_results.append(logo_df)

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    if skipped:
        pd.DataFrame(skipped).to_csv(OUT_DIR / "representation_skipped.csv", index=False)

    if not all_results:
        raise RuntimeError("No Stage C representations were available to evaluate.")

    results = pd.concat(all_results, ignore_index=True)
    summary = summarize_results(results)
    leaderboard = make_leaderboard(summary)

    results.to_csv(OUT_DIR / "representation_all_folds.csv", index=False)
    summary.to_csv(OUT_DIR / "representation_summary.csv", index=False)
    leaderboard.to_csv(OUT_DIR / "representation_leaderboard.csv", index=False)

    print("\nSaved:")
    print(f"- {OUT_DIR / 'representation_all_folds.csv'}")
    print(f"- {OUT_DIR / 'representation_summary.csv'}")
    print(f"- {OUT_DIR / 'representation_leaderboard.csv'}")
    if skipped:
        print(f"- {OUT_DIR / 'representation_skipped.csv'}")
    print("\nTop leaderboard rows:")
    print(leaderboard.head(12).to_string(index=False))


if __name__ == "__main__":
    main()
