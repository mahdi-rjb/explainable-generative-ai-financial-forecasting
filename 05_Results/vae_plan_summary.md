\# VAE for return sequences



The generative component of the system is a variational autoencoder that operates on sequences of daily log returns of the S and P five hundred.



The current design uses

\* input window length thirty days

\* a fully connected encoder and decoder

\* a latent space of dimension eight



The VAE is trained to reconstruct historical return sequences and to regularise the latent representation towards a standard normal distribution. After training, sampling from the latent space allows generation of synthetic return paths that resemble observed market behaviour. These synthetic paths will later be used in the dashboard to illustrate plausible future scenarios.





\## Synthetic return sequences



The trained VAE was used to sample synthetic thirty day windows of S and P five hundred log returns by drawing latent vectors from the standard normal prior and decoding them.



Empirically, the pooled distribution of synthetic returns has a mean and standard deviation that are close to the empirical distribution of real daily returns used for training. Histogram plots confirm that the synthetic series follow a similar shape, although some smoothing of extreme values remains visible. This behaviour is expected from a VAE and is sufficient for generating plausible scenario paths for the dashboard.





\## Scenario generation



To turn the VAE into a scenario generator, a single thirty day window of recent returns is encoded into the latent space. The latent mean around this window serves as a reference point. By sampling several latent vectors in a small neighborhood around this mean and decoding them, the model produces multiple synthetic return paths that share similar characteristics with the original window but differ in the exact sequence of returns. These paths will later be transformed to price scenarios and displayed in the dashboard.





