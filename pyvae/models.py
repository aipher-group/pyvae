import torch
import torch.nn as nn
import torch.nn.functional as F

from pyvae.layers import InformedLinear


class InformedVAE(nn.Module):
    def __init__(
        self,
        adj: torch.Tensor,
        latent_dim: int | None = None,
        seed: int = 42,
        l2_lambda: float = 1e-5,
    ):

        super().__init__()
        torch.manual_seed(seed)

        n_genes, n_pathways = adj.shape

        self.n_genes = n_genes
        self.n_pathways = n_pathways
        self.latent_dim = latent_dim if latent_dim is not None else n_pathways // 2
        self.l2_lambda = l2_lambda

        self.informed = InformedLinear(adj, activation="tanh")
        self.fc_mean = nn.Linear(self.n_pathways, self.latent_dim)
        self.fc_log_var = nn.Linear(self.n_pathways, self.latent_dim)
        self.dec_latent = nn.Linear(self.latent_dim, self.n_pathways)
        self.dec_out = nn.Linear(self.n_pathways, self.n_genes)

    def encode(self, x: torch.Tensor):
        h = self.informed(x)
        mu = self.fc_mean(h)
        log_var = self.fc_log_var(h)
        return mu, log_var, h

    def reparameterise(self, mu: torch.Tensor, log_var: torch.Tensor) -> torch.Tensor:
        clamped = torch.clamp(log_var, -3, 3)
        sigma = torch.exp(0.5 * clamped) * 0.01
        eps = torch.randn_like(sigma)
        return mu + sigma * eps

    def decode(self, z: torch.Tensor) -> torch.Tensor:

        h_prime = torch.tanh(self.dec_latent(z))
        x_hat = self.dec_out(h_prime)
        return x_hat

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

        recon_loss = F.mse_loss(recon, x) * self.n_genes
        kl_loss = -0.5 * torch.mean(1 + log_var - mu**2 - torch.exp(log_var))
        l2_loss = self.l2_lambda * torch.mean(h**2)
        return recon_loss + kl_loss + l2_loss
