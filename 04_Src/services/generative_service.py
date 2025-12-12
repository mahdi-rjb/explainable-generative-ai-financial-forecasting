from pathlib import Path
from typing import Tuple
import sys

import numpy as np
import pandas as pd
import torch

# Ensure 04_Src is on sys.path when this file is run directly
this_file = Path(__file__).resolve()
src_root = this_file.parents[1]  # .../04_Src
if str(src_root) not in sys.path:
    sys.path.insert(0, str(src_root))

from models.generative_vae import ReturnsVAE, generate_scenarios_from_window


def load_trained_vae(
    project_root: Path | None = None,
    seq_len: int = 30,
    latent_dim: int = 12,
    hidden_dim: int = 128,
) -> ReturnsVAE:

    """
    Load the trained VAE for return sequences from checkpoint.
    """
    if project_root is None:
        project_root = Path(__file__).resolve().parents[2]

    checkpoints_dir = project_root / "04_Src" / "models" / "checkpoints"
    vae_path = checkpoints_dir / "returns_vae.pt"

    model = ReturnsVAE(seq_len=seq_len, latent_dim=latent_dim, hidden_dim=hidden_dim)
    print("Loading VAE with latent_dim", latent_dim, "hidden_dim", hidden_dim)
    state_dict = torch.load(vae_path, map_location="cpu")
    model.load_state_dict(state_dict)
    model.eval()

    return model


def load_recent_return_window(
    project_root: Path | None = None,
    window_size: int = 30,
) -> Tuple[np.ndarray, pd.Timestamp]:
    """
    Load the most recent window of log returns from the daily features file.
    """
    if project_root is None:
        project_root = Path(__file__).resolve().parents[2]

    processed_dir = project_root / "02_Data" / "Processed"
    features_path = processed_dir / "sp500_features_daily.csv"

    df = pd.read_csv(features_path, parse_dates=["Date"])
    df = df.sort_values("Date").reset_index(drop=True)

    if len(df) < window_size:
        raise ValueError(
            f"Not enough rows {len(df)} for window size {window_size}"
        )

    df_window = df.iloc[-window_size:]
    last_date = df_window["Date"].iloc[-1]

    if "log_return" not in df_window.columns:
        raise ValueError("Expected a log_return column")

    window = df_window["log_return"].values

    return window, last_date


def generate_return_scenarios(
    model: ReturnsVAE,
    last_window: np.ndarray,
    num_paths: int = 20,
    noise_scale: float = 1.0,
) -> np.ndarray:
    """
    Generate synthetic return scenarios around a recent window.

    Returns
    -------
    np.ndarray
        Array of shape (num_paths, seq_len).
    """
    last_window_t = torch.tensor(last_window, dtype=torch.float32)
    scenarios_t = generate_scenarios_from_window(
        model=model,
        last_window=last_window_t,
        num_paths=num_paths,
        noise_scale=noise_scale,
    )
    return scenarios_t.cpu().numpy()
