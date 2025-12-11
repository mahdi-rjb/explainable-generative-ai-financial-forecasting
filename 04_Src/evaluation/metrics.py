from typing import Dict

import numpy as np


def regression_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> Dict[str, float]:
    """
    Compute basic regression metrics:
    mean absolute error
    root mean squared error
    directional accuracy sign agreement
    """
    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)

    mae = float(np.mean(np.abs(y_true - y_pred)))
    rmse = float(np.sqrt(np.mean((y_true - y_pred) ** 2)))

    # Directional accuracy: how often sign of prediction matches sign of true
    sign_true = np.sign(y_true)
    sign_pred = np.sign(y_pred)

    non_zero_mask = sign_true != 0
    if np.any(non_zero_mask):
        dir_acc = float(np.mean(sign_true[non_zero_mask] == sign_pred[non_zero_mask]))
    else:
        dir_acc = float("nan")

    return {
        "mae": mae,
        "rmse": rmse,
        "directional_accuracy": dir_acc,
    }


def format_metrics(metrics: Dict[str, float], prefix: str = "") -> str:
    """
    Format a metrics dictionary as a readable string.

    Parameters
    ----------
    metrics
        Dictionary with keys like mae, rmse, directional_accuracy.
    prefix
        Optional label to prepend.

    Returns
    -------
    str
        Human readable string.
    """
    parts = []
    for key in ["mae", "rmse", "directional_accuracy"]:
        if key in metrics:
            value = metrics[key]
            parts.append(f"{key}={value:.6f}")
    joined = ", ".join(parts)
    if prefix:
        return f"{prefix}: {joined}"
    return joined
