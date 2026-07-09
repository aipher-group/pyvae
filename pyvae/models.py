import torch
import torch.nn as nn

from pyvae.components import (
    DenseDecoder,
    Encoder,
    GaussianLikelihood,
    gaussian_kl,
    reparameterise,
)


class InformedVAE(nn.Module):
    def __init__(
        self,
        adj: torch.Tensor,
        latent_dim: int | None = None,
        seed: int = 42,
        l2_lambda: float = 1e-5,
        beta: float = 1.0,
    ):
        super().__init__()
        torch.manual_seed(seed)

        n_genes, n_pathways = adj.shape

        self.n_genes = n_genes
        self.n_pathways = n_pathways
        self.latent_dim = latent_dim if latent_dim is not None else n_pathways // 2
        self.l2_lambda = l2_lambda
        self.beta = beta

        self.encoder = Encoder(adj=adj, latent_dim=self.latent_dim)
        self.decoder = DenseDecoder(
            latent_dim=self.latent_dim,
            n_pathways=self.n_pathways,
            n_genes=self.n_genes,
        )
        self.likelihood = GaussianLikelihood()

    def encode(self, x: torch.Tensor):
        return self.encoder(x)

    def reparameterise(self, mu: torch.Tensor, log_var: torch.Tensor) -> torch.Tensor:
        return reparameterise(mu, log_var)

    def decode(self, z: torch.Tensor) -> torch.Tensor:
        return self.decoder(z)

    def forward(self, x: torch.Tensor):
        mu, log_var, h = self.encode(x)
        z = self.reparameterise(mu, log_var)
        recon = self.decode(z)
        return recon, mu, log_var, h

    def loss(
        self,
        x: torch.Tensor,
        recon: torch.Tensor,
        mu: torch.Tensor,
        log_var: torch.Tensor,
        h: torch.Tensor,
    ) -> torch.Tensor:
        recon_loss = self.likelihood(recon, x)
        kl_loss = gaussian_kl(mu, log_var)
        l2_loss = self.l2_lambda * (h**2).sum(dim=1).mean()
        return recon_loss + self.beta * kl_loss + l2_loss
