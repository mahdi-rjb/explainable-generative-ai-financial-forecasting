from pathlib import Path
from typing import Dict, Tuple
import sys

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
import shap

# Ensure 04_Src is on sys.path when this file is run directly
this_file = Path(__file__).resolve()
src_root = this_file.parents[1]
if str(src_root) not in sys.path:
    sys.path.insert(0, str(src_root))


def load_tabular_data(
    project_root: Path | None = None,
) -> Tuple[pd.DataFrame, pd.DataFrame, np.ndarray, np.ndarray, np.ndarray, np.ndarray, list]:
    """
    Load tabular train and validation sets and separate features and targets.
    """
    if project_root is None:
        project_root = Path(__file__).resolve().parents[2]

    processed_dir = project_root / "02_Data" / "Processed"
    train_path = processed_dir / "sp500_tabular_train.csv"
    val_path = processed_dir / "sp500_tabular_val.csv"

    df_train = pd.read_csv(train_path, parse_dates=["Date"])
    df_val = pd.read_csv(val_path, parse_dates=["Date"])

    target_col = "target_next_return"
    exclude_cols = ["Date", target_col]

    feature_cols = [c for c in df_train.columns if c not in exclude_cols]

    X_train = df_train[feature_cols].values
    y_train = df_train[target_col].values

    X_val = df_val[feature_cols].values
    y_val = df_val[target_col].values

    return df_train, df_val, X_train, y_train, X_val, y_val, feature_cols


def train_random_forest_for_explainability(
    X_train: np.ndarray,
    y_train: np.ndarray,
) -> RandomForestRegressor:
    """
    Train a random forest model used purely for explainability.
    """
    rf = RandomForestRegressor(
        n_estimators=200,
        max_depth=6,
        random_state=42,
        n_jobs=-1,
    )
    rf.fit(X_train, y_train)
    return rf


def compute_local_shap_for_index(
    rf: RandomForestRegressor,
    df_val: pd.DataFrame,
    X_val: np.ndarray,
    feature_cols: list,
    index: int,
) -> Dict[str, object]:
    """
    Compute a local SHAP explanation for one validation index.

    Returns a dictionary with:
    date, true_target, predicted_target, local_explanation_df
    """
    explainer = shap.TreeExplainer(rf)

    x_single = X_val[index : index + 1]
    shap_single = explainer.shap_values(x_single)[0]

    target_col = "target_next_return"

    date = df_val.iloc[index]["Date"]
    true_target = float(df_val.iloc[index][target_col])
    pred_target = float(rf.predict(x_single)[0])

    local_df = pd.DataFrame(
        {
            "feature": feature_cols,
            "value": x_single[0],
            "shap_value": shap_single,
        }
    ).sort_values("shap_value", ascending=False).reset_index(drop=True)

    return {
        "date": date,
        "true_target": true_target,
        "predicted_target": pred_target,
        "local_explanation": local_df,
    }
