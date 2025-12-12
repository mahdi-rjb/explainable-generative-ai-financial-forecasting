from pathlib import Path
from typing import Tuple

import numpy as np
import torch
import torch.nn as nn


class LSTMForecastingModel(nn.Module):
    """
    Simple LSTM based regression model for next day return forecasting.

    Input shape:
        (batch_size, seq_len, num_features)

    Output:
        (batch_size,)
    """

    """
    LSTM based regression model for next day return forecasting.

    The model expects input tensors of shape
        batch_size, seq_len, num_features

    and produces a single scalar prediction per sample,
    interpreted as the next day log return.
    """

    def __init__(
        self,
        num_features: int,
        hidden_size: int = 64,
        num_layers: int = 2,
        dropout: float = 0.1,
    ) -> None:
        super().__init__()

        self.num_features = num_features
        self.hidden_size = hidden_size
        self.num_layers = num_layers

        self.lstm = nn.LSTM(
            input_size=num_features,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0.0,
        )

        self.fc = nn.Linear(hidden_size, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass.

        Parameters
        ----------
        x
            Tensor of shape (batch_size, seq_len, num_features)

        Returns
        -------
        torch.Tensor
            Tensor of shape (batch_size,)
        """
        lstm_out, _ = self.lstm(x)
        last_hidden = lstm_out[:, -1, :]
        out = self.fc(last_hidden)
        return out.squeeze(-1)


def count_parameters(model: nn.Module) -> int:
    """
    Count trainable parameters in a model.
    """
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def example_forward_pass() -> None:
    """
    Run an example forward pass on random data.
    This is a sanity check that the model works.
    """
    seq_len = 30
    num_features = 8
    batch_size = 16

    model = LSTMForecastingModel(num_features=num_features)

    x_dummy = torch.randn(batch_size, seq_len, num_features)
    y_dummy = model(x_dummy)

    print("Dummy input shape:", x_dummy.shape)
    print("Dummy output shape:", y_dummy.shape)
    print("Number of parameters:", count_parameters(model))


if __name__ == "__main__":
    example_forward_pass()
