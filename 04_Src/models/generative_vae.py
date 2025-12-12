from typing import Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F


class ReturnsVAE(nn.Module):
    """
    Variational autoencoder for sequences of daily returns.

    Input and output shapes
        batch_size, seq_len

    The encoder compresses a sequence into a latent vector.
    The decoder reconstructs a sequence from a latent sample.
    """

    def __init__(
        self,
        seq_len: int,
        latent_dim: int = 8,
        hidden_dim: int = 64,
    ) -> None:
        super().__init__()

        self.seq_len = seq_len
        self.latent_dim = latent_dim
        self.hidden_dim = hidden_dim

        input_dim = seq_len  # one value per day

        # Encoder network
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
        )

        self.fc_mu = nn.Linear(hidden_dim, latent_dim)
        self.fc_logvar = nn.Linear(hidden_dim, latent_dim)

        # Decoder network
        self.decoder = nn.Sequential(
            nn.Linear(latent_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, input_dim),
        )

    def encode(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Encode input sequences into latent mean and log variance.

        x shape
            batch_size, seq_len
        """
        h = self.encoder(x)
        mu = self.fc_mu(h)
        logvar = self.fc_logvar(h)
        return mu, logvar

    @staticmethod
    def reparameterize(mu: torch.Tensor, logvar: torch.Tensor) -> torch.Tensor:
        """
        Reparameterisation trick to sample latent vector z.
        """
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std

    def decode(self, z: torch.Tensor) -> torch.Tensor:
        """
        Decode latent vector into reconstructed sequence.

        Output shape
            batch_size, seq_len
        """
        x_recon = self.decoder(z)
        return x_recon

    def forward(
        self, x: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Forward pass of the VAE.

        Returns reconstructed x, latent mean and latent log variance.

        x shape
            batch_size, seq_len
        """
        mu, logvar = self.encode(x)
        z = self.reparameterize(mu, logvar)
        x_recon = self.decode(z)
        return x_recon, mu, logvar


def vae_loss(
    x_recon: torch.Tensor,
    x_true: torch.Tensor,
    mu: torch.Tensor,
    logvar: torch.Tensor,
    beta: float = 1.0,
) -> torch.Tensor:
    """
    VAE loss with reconstruction term and Kullback Leibler divergence.

    Reconstruction loss
        mean squared error over all time steps

    KL term
        encourages latent distribution to be close to standard normal
    """
    recon_loss = F.mse_loss(x_recon, x_true, reduction="mean")

    # KL divergence between N(mu, sigma) and N(0, 1)
    kl_div = -0.5 * torch.mean(1 + logvar - mu.pow(2) - logvar.exp())

    return recon_loss + beta * kl_div


def example_forward_pass() -> None:
    """
    Run a small example to check that the model and loss work.
    """
    batch_size = 16
    seq_len = 30
    latent_dim = 8

    model = ReturnsVAE(seq_len=seq_len, latent_dim=latent_dim)

    x_dummy = torch.randn(batch_size, seq_len)
    x_recon, mu, logvar = model(x_dummy)

    loss = vae_loss(x_recon, x_dummy, mu, logvar)

    print("Dummy input shape:", x_dummy.shape)
    print("Reconstructed shape:", x_recon.shape)
    print("Latent mean shape:", mu.shape)
    print("Latent logvar shape:", logvar.shape)
    print("Example loss:", float(loss.item()))


def generate_scenarios_from_window(
    model: ReturnsVAE,
    last_window: torch.Tensor,
    num_paths: int,
    noise_scale: float = 1.0,
) -> torch.Tensor:
    """
    Generate multiple synthetic return windows conditioned loosely
    on one observed window.

    Parameters
    ----------
    model
        Trained ReturnsVAE instance in evaluation mode.
    last_window
        Tensor of shape (seq_len,) or (1, seq_len)
        representing one recent historical window of returns.
    num_paths
        Number of synthetic windows to generate.
    noise_scale
        Scale of additional Gaussian noise added in latent space.

    Returns
    -------
    torch.Tensor
        Tensor with shape (num_paths, seq_len) containing synthetic windows.
    """
    device = next(model.parameters()).device
    model.eval()

    if last_window.dim() == 1:
        x = last_window.unsqueeze(0)
    else:
        x = last_window

    x = x.to(device)

    with torch.no_grad():
        # Encode the reference window
        mu, logvar = model.encode(x)
        # Use only the mean as a base point in latent space
        base_z = mu[0]

        latent_dim = model.latent_dim

        # Sample around this base point
        z_samples = []
        for _ in range(num_paths):
            eps = torch.randn(latent_dim, device=device)
            z = base_z + noise_scale * eps
            z_samples.append(z)

        z_batch = torch.stack(z_samples, dim=0)
        x_synth = model.decode(z_batch)

    return x_synth


if __name__ == "__main__":
    example_forward_pass()
