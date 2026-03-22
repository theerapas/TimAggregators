import os
import warnings
import joblib
import numpy as np
import pandas as pd
from rdkit import Chem, RDLogger
from rdkit.Chem import Descriptors, rdFingerprintGenerator

from sklearn.base import clone
from sklearn.ensemble import (
    RandomForestClassifier,
    ExtraTreesClassifier,
    HistGradientBoostingClassifier,
)
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    matthews_corrcoef,
    roc_auc_score,
    average_precision_score,
)
from sklearn.model_selection import StratifiedKFold, LeaveOneGroupOut
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

# Optional models
HAVE_XGBOOST = False
HAVE_LIGHTGBM = False
HAVE_CATBOOST = False

try:
    from xgboost import XGBClassifier

    HAVE_XGBOOST = True
except Exception:
    XGBClassifier = None

try:
    from lightgbm import LGBMClassifier

    HAVE_LIGHTGBM = True
except Exception:
    LGBMClassifier = None

try:
    from catboost import CatBoostClassifier

    HAVE_CATBOOST = True
except Exception:
    CatBoostClassifier = None

warnings.filterwarnings("ignore")
RDLogger.DisableLog("rdApp.*")

# -----------------------------
# Config
# -----------------------------
DATA_DIR = "data"
SCREENING_FILE = os.path.join(DATA_DIR, "screening_data.tsv")
DRUGS_FILE = os.path.join(DATA_DIR, "selected_drugs_smiles.tsv")
EXCIPIENTS_FILE = os.path.join(DATA_DIR, "selected_excipients_smiles.tsv")

RANDOM_STATE = 42
THRESHOLDS = [0.2, 0.5]
N_SPLITS = 10

# Same chemistry setup as your current script
FP_RADIUS = 4
FP_SIZE = 2048


def load_data():
    print("Loading data...")
    screening_data = pd.read_csv(SCREENING_FILE, sep="\t")
    screening_data = screening_data[["DRUG", "EXCIPIENT", "CLASS"]].dropna(
        subset=["CLASS"]
    )

    drugs_smiles = pd.read_csv(DRUGS_FILE, sep="\t")
    excipients_smiles = pd.read_csv(EXCIPIENTS_FILE, sep="\t")

    print(f"Loaded {len(screening_data)} screening records.")
    print(f"Loaded {len(drugs_smiles)} selected drugs.")
    print(f"Loaded {len(excipients_smiles)} selected excipients.")
    return screening_data, drugs_smiles, excipients_smiles


def build_features(
    drugs_smiles: pd.DataFrame,
    excipients_smiles: pd.DataFrame,
    screening_data: pd.DataFrame,
):
    print("Extracting features...")
    descriptor_funcs = [func for _, func in Descriptors._descList]
    morgan_gen = rdFingerprintGenerator.GetMorganGenerator(
        radius=FP_RADIUS, fpSize=FP_SIZE
    )

    fp_feature_names = [f"fp_{i}" for i in range(FP_SIZE)]
    desc_feature_names = [name for name, _ in Descriptors._descList]
    single_feature_names = fp_feature_names + desc_feature_names

    def get_mol_features(smiles: str):
        total_len = FP_SIZE + len(descriptor_funcs)
        try:
            if pd.isna(smiles) or not isinstance(smiles, str) or not smiles.strip():
                return [np.nan] * total_len

            mol = Chem.MolFromSmiles(smiles)
            if mol is None:
                return [np.nan] * total_len

            fp = morgan_gen.GetFingerprint(mol)
            fp_features = list(fp)

            desc_features = []
            for func in descriptor_funcs:
                try:
                    v = float(func(mol))
                    if np.isnan(v) or np.isinf(v):
                        v = 0.0
                    desc_features.append(v)
                except Exception:
                    desc_features.append(0.0)

            features = np.array(fp_features + desc_features, dtype=np.float64)
            features = np.nan_to_num(features, nan=0.0, posinf=0.0, neginf=0.0)
            return features.tolist()
        except Exception:
            return [np.nan] * total_len

    print(f"Processing {len(drugs_smiles)} drugs...")
    drug_feature_df = pd.DataFrame(
        drugs_smiles["SMILES"].apply(get_mol_features).tolist(),
        columns=[f"Drug_{name}" for name in single_feature_names],
    )
    drugs_df = pd.concat([drugs_smiles[["NAME"]], drug_feature_df], axis=1)

    print(f"Processing {len(excipients_smiles)} excipients...")
    excipient_feature_df = pd.DataFrame(
        excipients_smiles["SMILES"].apply(get_mol_features).tolist(),
        columns=[f"Exc_{name}" for name in single_feature_names],
    )
    excipients_df = pd.concat(
        [excipients_smiles[["NAME"]], excipient_feature_df], axis=1
    )

    print("Merging features into training dataset...")
    dataset = pd.merge(
        screening_data,
        drugs_df,
        left_on="DRUG",
        right_on="NAME",
        how="left",
    ).drop(columns=["NAME"])

    dataset = pd.merge(
        dataset,
        excipients_df,
        left_on="EXCIPIENT",
        right_on="NAME",
        how="left",
    ).drop(columns=["NAME"])

    X = dataset.drop(columns=["DRUG", "EXCIPIENT", "CLASS"]).copy()
    X = X.replace([np.inf, -np.inf], np.nan).fillna(0.0).astype(np.float32)
    y = dataset["CLASS"].astype(int).copy()
    groups = dataset["DRUG"].copy()

    print(f"Final dataset shape for training: {X.shape}")
    return dataset, X, y, groups


def compute_metrics(y_true, y_pred, y_prob):
    result = {
        "MCC": matthews_corrcoef(y_true, y_pred),
        "F1": f1_score(y_true, y_pred, zero_division=0),
        "Precision": precision_score(y_true, y_pred, zero_division=0),
        "Accuracy": accuracy_score(y_true, y_pred),
        "Recall": recall_score(y_true, y_pred, zero_division=0),
    }

    unique_classes = np.unique(y_true)
    if len(unique_classes) < 2:
        result["AUROC"] = np.nan
        result["AUPRC"] = (
            np.nan if np.sum(y_true) == 0 else average_precision_score(y_true, y_prob)
        )
    else:
        result["AUROC"] = roc_auc_score(y_true, y_prob)
        result["AUPRC"] = average_precision_score(y_true, y_prob)

    return result


def get_probabilities(model, X):
    if hasattr(model, "predict_proba"):
        return model.predict_proba(X)[:, 1]

    if hasattr(model, "decision_function"):
        scores = model.decision_function(X)
        scores = np.asarray(scores, dtype=float)
        scores = (scores - scores.min()) / (scores.max() - scores.min() + 1e-12)
        return scores

    preds = model.predict(X)
    return np.asarray(preds, dtype=float)


def make_model_builders():
    def scale_pos_weight(y_train):
        pos = int(np.sum(y_train == 1))
        neg = int(np.sum(y_train == 0))
        return max(1.0, neg / max(pos, 1))

    builders = {
        "RandomForest": lambda y_train: RandomForestClassifier(
            n_estimators=500,
            random_state=RANDOM_STATE,
            n_jobs=-1,
        ),
        "ExtraTrees": lambda y_train: ExtraTreesClassifier(
            n_estimators=500,
            random_state=RANDOM_STATE,
            n_jobs=-1,
        ),
        "HistGradientBoosting": lambda y_train: HistGradientBoostingClassifier(
            max_iter=300,
            learning_rate=0.05,
            max_depth=8,
            random_state=RANDOM_STATE,
        ),
        "LogisticRegression": lambda y_train: Pipeline(
            [
                ("scaler", StandardScaler()),
                (
                    "model",
                    LogisticRegression(
                        max_iter=3000,
                        solver="liblinear",
                        random_state=RANDOM_STATE,
                    ),
                ),
            ]
        ),
    }

    if HAVE_XGBOOST:
        builders["XGBoost"] = lambda y_train: XGBClassifier(
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
            scale_pos_weight=scale_pos_weight(y_train),
        )

    if HAVE_LIGHTGBM:
        builders["LightGBM"] = lambda y_train: LGBMClassifier(
            n_estimators=500,
            learning_rate=0.05,
            num_leaves=31,
            subsample=0.8,
            colsample_bytree=0.8,
            objective="binary",
            class_weight="balanced",
            random_state=RANDOM_STATE,
            n_jobs=-1,
            verbose=-1,
        )

    if HAVE_CATBOOST:
        builders["CatBoost"] = lambda y_train: CatBoostClassifier(
            iterations=500,
            learning_rate=0.05,
            depth=6,
            loss_function="Logloss",
            eval_metric="AUC",
            random_seed=RANDOM_STATE,
            verbose=0,
        )

    return builders


def evaluate_stratified_cv(X, y, model_name, builder, thresholds):
    skf = StratifiedKFold(n_splits=N_SPLITS, shuffle=True, random_state=RANDOM_STATE)
    rows = []

    for fold, (train_idx, test_idx) in enumerate(skf.split(X, y), start=1):
        print(f"  [CV] {model_name} fold {fold}/{N_SPLITS}")

        X_train, X_test = X.iloc[train_idx], X.iloc[test_idx]
        y_train, y_test = y.iloc[train_idx], y.iloc[test_idx]

        model = builder(y_train)
        model.fit(X_train, y_train)
        y_prob = get_probabilities(model, X_test)

        for threshold in thresholds:
            y_pred = (y_prob >= threshold).astype(int)
            metrics = compute_metrics(y_test, y_pred, y_prob)
            metrics.update(
                {
                    "Model": model_name,
                    "Fold": fold,
                    "Threshold": threshold,
                    "Evaluation": "CV",
                }
            )
            rows.append(metrics)

    return pd.DataFrame(rows)


def evaluate_leave_one_drug_out(X, y, groups, model_name, builder, thresholds):
    logo = LeaveOneGroupOut()
    rows = []
    total_folds = groups.nunique()

    for fold, (train_idx, test_idx) in enumerate(
        logo.split(X, y, groups=groups), start=1
    ):
        held_out_drug = groups.iloc[test_idx].iloc[0]
        print(
            f"  [LOGO] {model_name} fold {fold}/{total_folds} - held out: {held_out_drug}"
        )

        X_train, X_test = X.iloc[train_idx], X.iloc[test_idx]
        y_train, y_test = y.iloc[train_idx], y.iloc[test_idx]

        model = builder(y_train)
        model.fit(X_train, y_train)
        y_prob = get_probabilities(model, X_test)

        for threshold in thresholds:
            y_pred = (y_prob >= threshold).astype(int)
            metrics = compute_metrics(y_test, y_pred, y_prob)
            metrics.update(
                {
                    "Model": model_name,
                    "HeldOutDrug": held_out_drug,
                    "Fold": fold,
                    "Threshold": threshold,
                    "Evaluation": "LOGO",
                }
            )
            rows.append(metrics)

    return pd.DataFrame(rows)


def summarize_results(df: pd.DataFrame):
    metric_cols = ["MCC", "F1", "Precision", "Accuracy", "Recall", "AUROC", "AUPRC"]
    summary = (
        df.groupby(["Evaluation", "Model", "Threshold"])[metric_cols]
        .agg(["mean", "std"])
        .round(4)
    )
    summary.columns = [f"{metric}_{stat}" for metric, stat in summary.columns]
    return summary.reset_index()


screening_data, drugs_smiles, excipients_smiles = load_data()
dataset, X, y, groups = build_features(drugs_smiles, excipients_smiles, screening_data)

builders = make_model_builders()
print("\nModels to compare:")
for name in builders:
    print(f"- {name}")

cv_all = []
logo_all = []

for model_name, builder in builders.items():
    print(f"\nRunning model: {model_name}")
    cv_df = evaluate_stratified_cv(X, y, model_name, builder, THRESHOLDS)
    logo_df = evaluate_leave_one_drug_out(X, y, groups, model_name, builder, THRESHOLDS)
    cv_all.append(cv_df)
    logo_all.append(logo_df)

cv_results = pd.concat(cv_all, ignore_index=True)
logo_results = pd.concat(logo_all, ignore_index=True)
all_results = pd.concat([cv_results, logo_results], ignore_index=True)

cv_summary = summarize_results(cv_results)
logo_summary = summarize_results(logo_results)
overall_summary = summarize_results(all_results)

# Threshold-independent ranking is usually most reliable for model comparison
leaderboard = overall_summary.sort_values(
    by=["Evaluation", "Threshold", "AUPRC_mean", "AUROC_mean", "MCC_mean"],
    ascending=[True, True, False, False, False],
).reset_index(drop=True)

os.makedirs("compare_model_results",exist_ok=True)
overall_summary.to_csv(
    "compare_model_results/multi_model_overall_summary.csv", index=False
)
leaderboard.to_csv("compare_model_results/multi_model_leaderboard.csv", index=False)

print("\nSaved:")
print("- multi_model_overall_summary.csv")
print("- multi_model_leaderboard.csv")

print("\nTop rows of leaderboard:")
print(leaderboard.head(12).to_string(index=False))
