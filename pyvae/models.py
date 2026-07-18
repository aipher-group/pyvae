import torch
import torch.nn as nn

from pyvae.components import (
    CountDecoder,
    DenseDecoder,
    Encoder,
    GaussianLikelihood,
    gaussian_kl,
    nb_reconstruction_loss,
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
        likelihood: str = "gaussian",
        n_cov: int = 0,
    ):
        super().__init__()
        torch.manual_seed(seed)

        n_genes, n_pathways = adj.shape

        self.n_genes = n_genes
        self.n_pathways = n_pathways
        self.latent_dim = latent_dim if latent_dim is not None else n_pathways // 2
        self.l2_lambda = l2_lambda
        self.beta = beta
        self.likelihood_kind = likelihood
        self.n_cov = n_cov

        self.encoder = Encoder(adj=adj, latent_dim=self.latent_dim, n_cov=n_cov)

        if likelihood == "gaussian":
            self.decoder = DenseDecoder(
                latent_dim=self.latent_dim,
                n_pathways=self.n_pathways,
                n_genes=self.n_genes,
                n_cov=n_cov,
            )
            self.likelihood = GaussianLikelihood()
        elif likelihood == "nb":
            self.decoder = CountDecoder(
                latent_dim=self.latent_dim,
                n_pathways=self.n_pathways,
                n_genes=self.n_genes,
                n_cov=n_cov,
            )
            self.likelihood = None
        else:
            raise ValueError(
                f"unknown likelihood: {likelihood!r} (expected 'gaussian' or 'nb')"
            )

    def encode(self, x: torch.Tensor, cov: torch.Tensor | None = None):
        return self.encoder(x, cov)

    def reparameterise(self, mu: torch.Tensor, log_var: torch.Tensor) -> torch.Tensor:
        return reparameterise(mu, log_var)

    def decode(self, z: torch.Tensor, cov: torch.Tensor | None = None) -> torch.Tensor:
        return self.decoder(z, cov)

    def forward(self, x: torch.Tensor, cov: torch.Tensor | None = None):
        mu, log_var, h = self.encode(x, cov)
        z = self.reparameterise(mu, log_var)
        recon = self.decode(z, cov)
        return recon, mu, log_var, h

    def loss(
        self,
        x: torch.Tensor,
        recon: torch.Tensor,
        mu: torch.Tensor,
        log_var: torch.Tensor,
        h: torch.Tensor,
        counts: torch.Tensor | None = None,
        library: torch.Tensor | None = None,
        beta: float | None = None,
    ) -> torch.Tensor:
        """x is ignored when likelihood_kind == "nb"; counts/library are used instead.

        beta : optional override for self.beta on this call only. When None
            (the default), self.beta is used. Passing an explicit value avoids
            mutating self.beta across epochs during KL warmup.
        """
        if self.likelihood_kind == "gaussian":
            recon_loss = self.likelihood(recon, x)
        else:  # nb
            if counts is None or library is None:
                raise ValueError(
                    "NB likelihood requires both 'counts' and 'library' kwargs"
                )
            recon_loss = nb_reconstruction_loss(
                counts, recon, library, self.decoder.theta
            )

        kl_loss = gaussian_kl(mu, log_var)
        l2_loss = self.l2_lambda * (h**2).sum(dim=1).mean()
        beta_eff = self.beta if beta is None else beta
        return recon_loss + beta_eff * kl_loss + l2_loss
