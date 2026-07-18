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

    @torch.no_grad()
    def predict_counterfactual(
        self,
        x: torch.Tensor,
        library: torch.Tensor,
        cov_from: torch.Tensor,
        cov_to: torch.Tensor,
    ) -> torch.Tensor:
        """Predict expression under a different covariate, holding latent biology fixed.

        Encodes ``x`` under its real covariate ``cov_from`` to obtain the
        posterior MEAN (mu) -- not a stochastic sample of z -- then decodes
        that mean under the swapped covariate ``cov_to``. Using the mean
        (rather than sampling) keeps the prediction deterministic: the only
        thing that changes between two calls with the same inputs is the
        covariate, not random noise from reparameterise().

        Requires ``likelihood_kind == "nb"``; raises otherwise since the
        return value is expressed in count space using ``library``.

        Parameters
        ----------
        x : (batch, n_genes) log1p-normalized expression of the real cells.
        library : (batch, 1) real per-cell library size to scale the prediction into.
        cov_from : (batch, n_cov) the cells' true one-hot covariate.
        cov_to : (batch, n_cov) the one-hot covariate to decode under instead.

        Returns
        -------
        predicted_counts : (batch, n_genes) predicted mean counts under cov_to
            (px_scale * library), directly comparable to real count profiles.
        """
        if self.likelihood_kind != "nb":
            raise ValueError(
                f"predict_counterfactual requires likelihood_kind == 'nb', "
                f"got {self.likelihood_kind!r}"
            )
        mu, _log_var, _h = self.encode(x, cov_from)
        px_scale = self.decode(mu, cov_to)
        return px_scale * library
