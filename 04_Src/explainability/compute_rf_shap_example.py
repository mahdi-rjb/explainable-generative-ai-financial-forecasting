import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
import shap


def load_tabular_sets(processed_dir: Path):
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


def main() -> None:
    project_root = Path(__file__).resolve().parents[2]
    processed_dir = project_root / "02_Data" / "Processed"
    results_dir = project_root / "05_Results"

    results_dir.mkdir(parents=True, exist_ok=True)

    df_train, df_val, X_train, y_train, X_val, y_val, feature_cols = load_tabular_sets(
        processed_dir
    )

    rf = RandomForestRegressor(
        n_estimators=200,
        max_depth=6,
        random_state=42,
        n_jobs=-1,
    )
    rf.fit(X_train, y_train)

    explainer = shap.TreeExplainer(rf)

    # Example index: last row in validation
    idx = len(df_val) - 1

    x_single = X_val[idx : idx + 1]
    y_true_single = float(y_val[idx])
    y_pred_single = float(rf.predict(x_single)[0])
    date_single = df_val.iloc[idx]["Date"].strftime("%Y-%m-%d")

    shap_single = explainer.shap_values(x_single)[0]

    df_local = pd.DataFrame(
        {
            "feature": feature_cols,
            "value": x_single[0],
            "shap_value": shap_single,
        }
    ).sort_values("shap_value", ascending=False)

    local_csv_path = results_dir / "rf_shap_example_local_explanation_script.csv"
    df_local.to_csv(local_csv_path, index=False)

    meta = {
        "date": date_single,
        "y_true": y_true_single,
        "y_pred": y_pred_single,
        "csv_path": str(local_csv_path),
    }
    meta_path = results_dir / "rf_shap_example_meta.json"
    with open(meta_path, "w", encoding="utf8") as f:
        json.dump(meta, f, indent=2)

    print("Saved local SHAP explanation to:", local_csv_path)
    print("Saved meta information to:", meta_path)


if __name__ == "__main__":
    main()
