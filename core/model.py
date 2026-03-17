"""
core/model.py
-------------
Single source of truth for XGBoost model config and training logic.

Every script that trains or uses the model imports from here:
    from core.model import make_model, train_model

Never define XGBClassifier config locally in backtest.py, train.py,
or predict.py. Change hyperparameters here and they apply everywhere.
"""

import numpy as np
import pandas as pd
from xgboost import XGBClassifier
from sklearn.metrics import roc_auc_score


def make_model() -> XGBClassifier:
    """
    Build the XGBoost classifier with the current best hyperparameters.

    Key design choices:
    - max_depth=2: shallow trees prevent overfitting on ~3000 rows
    - heavy regularisation (reg_alpha, reg_lambda, min_child_weight):
      keeps the model from learning noise in small folds
    - early_stopping_rounds=50: halts training when val loss stops improving
      so n_estimators=1000 is a ceiling, not a fixed count
    - no StandardScaler needed: trees split on thresholds not magnitudes
    """
    return XGBClassifier(
        n_estimators       = 1000,
        max_depth          = 2,
        learning_rate      = 0.01,
        subsample          = 0.6,
        colsample_bytree   = 0.4,
        min_child_weight   = 20,
        reg_alpha          = 2.0,
        reg_lambda         = 5.0,
        early_stopping_rounds = 50,
        eval_metric        = "logloss",
        random_state       = 42,
    )


def train_model(
    X_train: pd.DataFrame,
    y_train: pd.Series,
    X_val: pd.DataFrame,
    y_val: pd.Series,
    verbose: bool = False,
) -> XGBClassifier:
    """
    Train the model on training data with early stopping on validation set.

    Args:
        X_train: training features
        y_train: training labels
        X_val:   validation features (used for early stopping only, not tuning)
        y_val:   validation labels
        verbose: print tree-by-tree progress (default False)

    Returns:
        Fitted XGBClassifier
    """
    model = make_model()
    model.fit(
        X_train, y_train,
        eval_set=[(X_val, y_val)],
        verbose=verbose,
    )
    return model


def train_model_full(
    X_train: pd.DataFrame,
    y_train: pd.Series,
    verbose: bool = False,
) -> XGBClassifier:
    """
    Train on all available data with no early stopping.

    Used by predict.py when training on the full historical dataset
    before generating live predictions. No validation set is needed
    because we are not selecting hyperparameters — we use all n_estimators.

    Args:
        X_train: all historical features
        y_train: all historical labels
        verbose: print progress (default False)

    Returns:
        Fitted XGBClassifier
    """
    model = XGBClassifier(
        n_estimators       = 1000,
        max_depth          = 2,
        learning_rate      = 0.01,
        subsample          = 0.6,
        colsample_bytree   = 0.4,
        min_child_weight   = 20,
        reg_alpha          = 2.0,
        reg_lambda         = 5.0,
        eval_metric        = "logloss",
        random_state       = 42,
        # No early_stopping_rounds — we use all estimators on full data
    )
    model.fit(X_train, y_train, verbose=verbose)
    return model


def get_feature_importance(
    model: XGBClassifier,
    feature_names: list,
    top_n: int = 15,
) -> pd.DataFrame:
    """
    Return a sorted DataFrame of feature importances.

    Useful for debugging which features are driving predictions.

    Args:
        model:         fitted XGBClassifier
        feature_names: list of feature column names
        top_n:         how many top features to return

    Returns:
        DataFrame with columns: feature, importance
    """
    importances = model.feature_importances_
    df = pd.DataFrame({
        "feature":    feature_names,
        "importance": importances,
    }).sort_values("importance", ascending=False)

    return df.head(top_n).reset_index(drop=True)


def evaluate_auc(
    model: XGBClassifier,
    X: pd.DataFrame,
    y: pd.Series,
) -> float:
    """
    Calculate AUC for a fitted model on a given dataset.

    Returns float AUC, or nan if calculation fails
    (e.g. only one class present in y).
    """
    try:
        probs = model.predict_proba(X)[:, 1]
        return roc_auc_score(y, probs)
    except Exception:
        return float("nan")