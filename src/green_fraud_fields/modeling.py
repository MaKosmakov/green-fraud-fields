from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
from lightgbm import LGBMClassifier, early_stopping
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.metrics import (
    average_precision_score,
    brier_score_loss,
    roc_auc_score,
    roc_curve,
)
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder


def make_preprocessor(frame: pd.DataFrame) -> ColumnTransformer:
    categorical = [c for c in frame if str(frame[c].dtype) in {"object", "string", "category"}]
    numeric = [c for c in frame if c not in categorical]
    return ColumnTransformer([
        ("numeric", SimpleImputer(strategy="median", add_indicator=True), numeric),
        ("categorical", Pipeline([
            ("impute", SimpleImputer(strategy="most_frequent")),
            ("encode", OneHotEncoder(handle_unknown="ignore", min_frequency=20)),
        ]), categorical),
    ])


def alert_metrics(y_true: np.ndarray, prediction: np.ndarray, budget: float) -> dict[str, float]:
    count = max(1, int(len(prediction) * budget))
    chosen = np.argsort(-prediction)[:count]
    positives = float(np.sum(y_true))
    hits = float(np.sum(y_true[chosen]))
    return {
        f"precision_at_{budget:g}": hits / count,
        f"recall_at_{budget:g}": hits / positives if positives else np.nan,
    }


def evaluate(y_true: np.ndarray, prediction: np.ndarray) -> dict[str, float]:
    fpr, tpr, _ = roc_curve(y_true, prediction)
    eligible = tpr[fpr <= 0.005]
    bins = np.linspace(0.0, 1.0, 11)
    bin_id = np.clip(np.digitize(prediction, bins) - 1, 0, 9)
    ece = 0.0
    for index in range(10):
        mask = bin_id == index
        if mask.any():
            ece += mask.mean() * abs(float(y_true[mask].mean() - prediction[mask].mean()))
    metrics = {
        "prevalence": float(np.mean(y_true)),
        "auc_pr": float(average_precision_score(y_true, prediction)),
        "roc_auc": float(roc_auc_score(y_true, prediction)),
        "brier": float(brier_score_loss(y_true, prediction)),
        "ece_10": float(ece),
        "recall_at_fpr_0.005": float(np.max(eligible)) if eligible.size else 0.0,
    }
    for budget in (0.005, 0.01, 0.02, 0.05):
        metrics.update(alert_metrics(y_true, prediction, budget))
    return metrics


def fit_lightgbm(
    x_train: pd.DataFrame,
    y_train: np.ndarray,
    x_valid: pd.DataFrame,
    y_valid: np.ndarray,
    seed: int,
):
    preprocessor = make_preprocessor(x_train)
    transformed_train = preprocessor.fit_transform(x_train)
    transformed_valid = preprocessor.transform(x_valid)
    positive = max(float(np.sum(y_train)), 1.0)
    scale = (len(y_train) - positive) / positive
    model = LGBMClassifier(
        n_estimators=500,
        learning_rate=0.05,
        num_leaves=31,
        max_depth=-1,
        subsample=0.9,
        colsample_bytree=0.9,
        reg_lambda=1.0,
        scale_pos_weight=scale,
        random_state=seed,
        n_jobs=-1,
        verbosity=-1,
    )
    model.fit(
        transformed_train,
        y_train,
        eval_set=[(transformed_valid, y_valid)],
        eval_metric="average_precision",
        callbacks=[early_stopping(40, verbose=False)],
    )
    return preprocessor, model


def save_json(path: str | Path, payload: dict) -> None:
    Path(path).write_text(json.dumps(payload, indent=2, allow_nan=True), encoding="utf-8")

