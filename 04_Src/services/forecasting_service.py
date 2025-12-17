from pathlib import Path
from typing import Dict, Tuple
import numpy as np
import pandas as pd
import torch
import sys

# Ensure 04_Src is on sys.path when this file is run directly
this_file = Path(__file__).resolve()
src_root = this_file.parents[1]  # .../04_Src
if str(src_root) not in sys.path:
    sys.path.insert(0, str(src_root))

from models.forecasting_model import LSTMForecastingModel


def load_lstm_model_and_metadata(
    project_root: Path | None = None,
) -> Tuple[LSTMForecastingModel, dict]:
    """
    Load the trained LSTM model and associated metadata
    using the sequence data that includes VIX.
    """
    if project_root is None:
        project_root = Path(__file__).resolve().parents[2]

    processed_dir = project_root / "02_Data" / "Processed"
    models_dir = project_root / "04_Src" / "models" / "checkpoints"

    data_path = processed_dir / "sp500_model_data_window30_with_vix.npz"
    checkpoint_path = models_dir / "lstm_forecasting_with_vix.pt"

    npz = np.load(data_path, allow_pickle=True)

    X_train = npz["X_train"]
    feature_cols = [str(c) for c in npz["feature_cols"]]
    window_size = int(npz["window_size"])

    num_features = X_train.shape[2]

    model = LSTMForecastingModel(
        num_features=num_features,
        hidden_size=64,
        num_layers=2,
        dropout=0.1,
    )
    state_dict = torch.load(checkpoint_path, map_location="cpu")
    model.load_state_dict(state_dict)
    model.eval()

    meta = {
        "window_size": window_size,
        "feature_cols": feature_cols,
    }

    return model, meta



def load_latest_feature_window(
    project_root: Path | None,
    window_size: int,
) -> tuple[np.ndarray, pd.Timestamp]:
    """
    Load the most recent window of features for forecasting from
    the daily features file that includes VIX.
    """
    if project_root is None:
        project_root = Path(__file__).resolve().parents[2]

    processed_dir = project_root / "02_Data" / "Processed"
    features_path = processed_dir / "sp500_features_daily_with_vix.csv"

    df = pd.read_csv(features_path, parse_dates=["Date"])
    df = df.sort_values("Date").reset_index(drop=True)

    if len(df) < window_size:
        raise ValueError(
            f"Not enough rows {len(df)} for window size {window_size}"
        )

    df_window = df.iloc[-window_size:]
    last_date = df_window["Date"].iloc[-1]

    # feature columns must match the ones used for training
    # we reuse the same logic as in the sequence builder
    exclude_cols = ["Date"]
    feature_cols = [c for c in df_window.columns if c not in exclude_cols and c != "target_next_return"]

    window = df_window[feature_cols].values.astype("float32")

    return window, last_date


def predict_next_return_with_lstm(
    model: LSTMForecastingModel,
    feature_window: np.ndarray,
) -> float:
    """
    Use the trained LSTM model to predict the next day log return
    from one feature window.

    Parameters
    ----------
    model
        Trained LSTMForecastingModel in eval mode.
    feature_window
        Array of shape (window_size, num_features).

    Returns
    -------
    float
        Predicted next day log return.
    """
    model.eval()
    with torch.no_grad():
        x = torch.tensor(
            feature_window[np.newaxis, :, :],
            dtype=torch.float32,
        )
        y_pred = model(x)
        return float(y_pred.cpu().numpy().reshape(-1)[0])

if __name__ == "__main__":
    project_root = Path(__file__).resolve().parents[2]

    model, meta = load_lstm_model_and_metadata(project_root=project_root)
    window, last_date = load_latest_feature_window(
        project_root=project_root,
        window_size=meta["window_size"],
    )
    pred = predict_next_return_with_lstm(model, window)

    print("Self test of forecasting_service")
    print("Last date in window:", last_date)
    print("Predicted next day log return:", pred)
