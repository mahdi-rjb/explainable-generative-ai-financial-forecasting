from pathlib import Path

import numpy as np
import pandas as pd


def main() -> None:
    project_root = Path(__file__).resolve().parents[2]
    processed_dir = project_root / "02_Data" / "Processed"

    features_path = processed_dir / "sp500_features_daily_with_vix.csv"
    output_path = processed_dir / "sp500_model_data_window30_with_vix.npz"

    print(f"Loading features from: {features_path}")
    df = pd.read_csv(features_path, parse_dates=["Date"])
    df = df.sort_values("Date").reset_index(drop=True)

    if "log_return" not in df.columns:
        raise ValueError("Expected a log_return column in the features file")

    # create next day target
    df["target_next_return"] = df["log_return"].shift(-1)
    df = df.dropna(subset=["target_next_return"]).reset_index(drop=True)

    # select feature columns
    exclude_cols = ["Date", "target_next_return"]
    feature_cols = [c for c in df.columns if c not in exclude_cols]

    print("Number of features:", len(feature_cols))
    print("Some features:", feature_cols[:10])

    X_all = df[feature_cols].values.astype("float32")
    y_all = df["target_next_return"].values.astype("float32")
    dates_all = df["Date"].values

    window_size = 30

    X_list = []
    y_list = []

    # we also keep one date per window, usually the target day
    date_list = []

    for i in range(len(df) - window_size):
        window = X_all[i : i + window_size]
        target = y_all[i + window_size]
        date_target = dates_all[i + window_size]

        X_list.append(window)
        y_list.append(target)
        date_list.append(date_target)

    X = np.stack(X_list, axis=0)
    y = np.array(y_list, dtype="float32")
    dates_seq = np.array(date_list)

    print("X shape:", X.shape)
    print("y shape:", y.shape)
    print("dates shape:", dates_seq.shape)

    # simple chronological split
    n = len(X)
    n_train = int(n * 0.7)
    n_val = int(n * 0.15)
    n_test = n - n_train - n_val

    X_train = X[:n_train]
    y_train = y[:n_train]
    dates_train = dates_seq[:n_train]

    X_val = X[n_train : n_train + n_val]
    y_val = y[n_train : n_train + n_val]
    dates_val = dates_seq[n_train : n_train + n_val]

    X_test = X[n_train + n_val :]
    y_test = y[n_train + n_val :]
    dates_test = dates_seq[n_train + n_val :]

    print("Train:", X_train.shape, "Val:", X_val.shape, "Test:", X_test.shape)
    print("Train dates:", dates_train[0], "to", dates_train[-1])
    print("Val dates:", dates_val[0], "to", dates_val[-1])
    print("Test dates:", dates_test[0], "to", dates_test[-1])

    np.savez(
        output_path,
        X_train=X_train,
        y_train=y_train,
        X_val=X_val,
        y_val=y_val,
        X_test=X_test,
        y_test=y_test,
        dates_train=dates_train,
        dates_val=dates_val,
        dates_test=dates_test,
        feature_cols=np.array(feature_cols),
        window_size=window_size,
    )

    print(f"Saved sequence data with VIX to: {output_path}")


if __name__ == "__main__":
    main()
