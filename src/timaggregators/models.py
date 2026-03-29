import numpy as np
from sklearn.ensemble import (
    RandomForestClassifier,
    ExtraTreesClassifier,
    HistGradientBoostingClassifier,
)
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

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

def make_model_builders(random_state=42):
    def scale_pos_weight(y_train):
        if y_train is None:
            return 1.0
        pos = int(np.sum(y_train == 1))
        neg = int(np.sum(y_train == 0))
        return max(1.0, neg / max(pos, 1))

    builders = {
        "RandomForest": lambda y_train=None: RandomForestClassifier(
            n_estimators=500,
            random_state=random_state,
            n_jobs=-1,
        ),
        "ExtraTrees": lambda y_train=None: ExtraTreesClassifier(
            n_estimators=500,
            random_state=random_state,
            n_jobs=-1,
        ),
        "HistGradientBoosting": lambda y_train=None: HistGradientBoostingClassifier(
            max_iter=300,
            learning_rate=0.05,
            max_depth=8,
            random_state=random_state,
        ),
        "LogisticRegression": lambda y_train=None: Pipeline(
            [
                ("scaler", StandardScaler()),
                (
                    "model",
                    LogisticRegression(
                        max_iter=3000,
                        solver="liblinear",
                        random_state=random_state,
                    ),
                ),
            ]
        ),
    }

    if HAVE_XGBOOST:
        builders["XGBoost"] = lambda y_train=None: XGBClassifier(
            n_estimators=500,
            learning_rate=0.05,
            max_depth=6,
            min_child_weight=1,
            subsample=0.8,
            colsample_bytree=0.8,
            reg_lambda=1.0,
            objective="binary:logistic",
            eval_metric="logloss",
            random_state=random_state,
            n_jobs=-1,
            tree_method="hist",
            scale_pos_weight=scale_pos_weight(y_train),
        )

    if HAVE_LIGHTGBM:
        builders["LightGBM"] = lambda y_train=None: LGBMClassifier(
            n_estimators=500,
            learning_rate=0.05,
            num_leaves=31,
            subsample=0.8,
            colsample_bytree=0.8,
            objective="binary",
            class_weight="balanced",
            random_state=random_state,
            n_jobs=-1,
            verbose=-1,
        )

    if HAVE_CATBOOST:
        builders["CatBoost"] = lambda y_train=None: CatBoostClassifier(
            iterations=500,
            learning_rate=0.05,
            depth=6,
            loss_function="Logloss",
            eval_metric="AUC",
            random_seed=random_state,
            verbose=0,
        )

    return builders
