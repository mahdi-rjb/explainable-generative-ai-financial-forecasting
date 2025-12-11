import os
from pathlib import Path
from typing import Dict, Tuple

import numpy as np
import pandas as pd


def load_feature_data(path: Path) -> pd.DataFrame:
    """
    Load merged daily feature data for S and P five hundred.

    Expected columns include:
    - Date
    - log_return
    - technical and sentiment features.
    """
    if not path.exists():
        raise FileNotFoundError(f"Feature file not found at {path}")

    df = pd.read_csv(path)

    if "Date" not in df.columns:
        raise ValueError("Expected a 'Date' column in the feature file")
    if "log_return" not in df.columns:
        raise ValueError("Expected a 'log_return' column in the feature file")

    df["Date"] = pd.to_datetime(df["Date"])

    # Sort just to be safe
    df = df.sort_values("Date").reset_index(drop=True)

    return df


def select_feature_columns(df: pd.DataFrame) -> Tuple[pd.DataFrame, list]:
    """
    Select feature columns for modeling.

    We exclude:
    - Date
    and keep all other numeric columns as features.
    """
    # Exclude obvious non features
    exclude_cols = {"Date"}

    feature_cols = [col for col in df.columns if col not in exclude_cols]

    # Target is log_return, but for the input features we can still keep it,
    # or we can remove it if we want strict separation.
    # For now we keep it as a feature as well.
    X_df = df[feature_cols].copy()

    return X_df, feature_cols


def build_sequences(
    df: pd.DataFrame,
    window_size: int = 30,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, list]:
    """
    Build sequence data for forecasting.

    For each day t (starting from window_size),
    we use the previous 'window_size' days of features to predict log_return at day t.

    Returns
    -------
    X : np.ndarray
        Shape (num_samples, window_size, num_features)
    y : np.ndarray
        Shape (num_samples,)
    target_dates : np.ndarray
        Shape (num_samples,)
        The dates corresponding to the target y values.
    feature_cols : list
        Names of the feature columns.
    """
    df = df.copy()

    # Separate target and features
    if "log_return" not in df.columns:
        raise ValueError("Expected 'log_return' column in df")

    target = df["log_return"].values
    dates = df["Date"].values

    X_df, feature_cols = select_feature_columns(df)
    feature_values = X_df.values

    num_rows, num_features = feature_values.shape

    if num_rows <= window_size:
        raise ValueError(
            f"Not enough rows ({num_rows}) for window_size={window_size}"
        )

    X_list = []
    y_list = []
    date_list = []

    # At index t, we predict log_return at t using the previous window_size days
    for t in range(window_size, num_rows):
        window_start = t - window_size
        window_end = t

        window = feature_values[window_start:window_end, :]
        X_list.append(window)
        y_list.append(target[t])
        date_list.append(dates[t])

    X = np.stack(X_list)  # (num_samples, window_size, num_features)
    y = np.array(y_list)
    target_dates = np.array(date_list)

    return X, y, target_dates, feature_cols


def split_by_date(
    X: np.ndarray,
    y: np.ndarray,
    dates: np.ndarray,
    train_end: pd.Timestamp,
    val_end: pd.Timestamp,
) -> Dict[str, np.ndarray]:
    """
    Split sequence data into train, validation and test sets
    based on the target date.

    Parameters
    ----------
    X : np.ndarray
        Shape (num_samples, window_size, num_features)
    y : np.ndarray
        Shape (num_samples,)
    dates : np.ndarray
        Shape (num_samples,)
        Target dates.
    train_end : pd.Timestamp
        Last date included in the training set.
    val_end : pd.Timestamp
        Last date included in the validation set.
        Test set uses dates after val_end.

    Returns
    -------
    dict
        Dictionary with keys:
        - X_train, y_train
        - X_val, y_val
        - X_test, y_test
        - dates_train, dates_val, dates_test
    """
    # Convert to pandas datetime for comparison
    dates_pd = pd.to_datetime(dates)

    train_mask = dates_pd <= train_end
    val_mask = (dates_pd > train_end) & (dates_pd <= val_end)
    test_mask = dates_pd > val_end

    def subset(mask):
        return X[mask], y[mask], dates_pd[mask]

    X_train, y_train, dates_train = subset(train_mask)
    X_val, y_val, dates_val = subset(val_mask)
    X_test, y_test, dates_test = subset(test_mask)

    result = {
        "X_train": X_train,
        "y_train": y_train,
        "dates_train": dates_train.values,
        "X_val": X_val,
        "y_val": y_val,
        "dates_val": dates_val.values,
        "X_test": X_test,
        "y_test": y_test,
        "dates_test": dates_test.values,
    }

    return result


def save_npz(
    arrays: Dict[str, np.ndarray],
    feature_cols: list,
    path: Path,
    window_size: int,
) -> None:
    """
    Save arrays and metadata into a compressed npz file.
    """
    os.makedirs(path.parent, exist_ok=True)

    np.savez_compressed(
        path,
        window_size=window_size,
        feature_cols=np.array(feature_cols),
        **arrays,
    )
    print(f"Saved model arrays to: {path}")


def main() -> None:
    project_root = Path(__file__).resolve().parents[2]
    processed_dir = project_root / "02_Data" / "Processed"

    features_path = processed_dir / "sp500_features_daily.csv"
    output_path = processed_dir / "sp500_model_data_window30.npz"

    print(f"Loading features from: {features_path}")
    df = load_feature_data(features_path)

    window_size = 30

    print(f"Building sequences with window_size={window_size}")
    X, y, target_dates, feature_cols = build_sequences(df, window_size=window_size)

    print(f"Total samples: {X.shape[0]}, window_size: {X.shape[1]}, num_features: {X.shape[2]}")

    # Define date based splits
    train_end = pd.Timestamp("2022-12-31")
    val_end = pd.Timestamp("2023-12-31")

    print(f"Splitting with train_end={train_end.date()}, val_end={val_end.date()}")
    splits = split_by_date(X, y, target_dates, train_end=train_end, val_end=val_end)

    print("Split sizes:")
    print("Train:", splits["X_train"].shape[0])
    print("Val:  ", splits["X_val"].shape[0])
    print("Test: ", splits["X_test"].shape[0])

    save_npz(splits, feature_cols, output_path, window_size=window_size)


if __name__ == "__main__":
    main()
