"""
PyTorch InformedVAE - Variational Autoencoder with biological constraints.

Architecture overview:

    Input x  (n_genes,)
        |
    InformedLinear(adj, tanh)       <- constrained: only allowed gene->pathway
        |  h  (n_pathways,)
   fc_mean / fc_log_var             <- two separate heads, shape (latent_dim,)
        |
    reparameterise  z = mu + sigma*eps  <- differentiable sampling trick
        |
    dec_latent (tanh)  (n_pathways,)    <- unconstrained decoder
    dec_out   (linear) (n_genes,)
        |
    Reconstructed x_hat  (n_genes,)

Loss = MSE reconstruction + KL divergence + L2 on pathway activations h
"""

import torch
import torch.nn as nn
import torch.nn.functional as F  # noqa: F401 - students will use F.mse_loss

from pyvae.layers import InformedLinear  # noqa: F401 - students will use this in __init__


class InformedVAE(nn.Module):
    """
    Informed Variational Autoencoder in PyTorch.

    Parameters
    ----------
    adj : torch.Tensor, shape (n_genes, n_pathways)
        Binary adjacency matrix from Reactome / D2 hierarchy.
        Passed directly to InformedLinear.
    latent_dim : int, optional
        Bottleneck dimension z.  Defaults to n_pathways // 2.
        Smaller values force stronger compression.
    seed : int
        Manual seed for reproducibility (applied via torch.manual_seed).
    l2_lambda : float
        Weight of the L2 regularisation term on pathway activations h.
        Keeps pathway representations sparse and interpretable.

    Layers to create in __init__
    ----------------------------
    informed    : biologically-constrained encoder layer (genes -> pathways).
    fc_mean     : projects pathway activations to the latent mean.
    fc_log_var  : projects pathway activations to the latent log-variance.
    dec_latent  : first decoder layer (latent -> pathways), apply tanh in decode.
    dec_out     : output decoder layer (pathways -> genes), no activation.
    """

    def __init__(  # pylint: disable=too-many-instance-attributes
        self,
        adj: torch.Tensor,
        latent_dim: int | None = None,
        seed: int = 42,
        l2_lambda: float = 1e-5,
    ):
        """Initialise encoder, decoder layers and store hyperparameters."""
        super().__init__()
        torch.manual_seed(seed)

        n_genes, n_pathways = adj.shape  # provided for you

        # Store dimensions - you will need them in loss() and encode()
        self.n_genes = n_genes
        self.n_pathways = n_pathways
        self.latent_dim = latent_dim if latent_dim is not None else n_pathways // 2
        self.l2_lambda = l2_lambda

        raise NotImplementedError

    # -------------------------------------------------------------------

    def encode(self, x: torch.Tensor):
        """
        Run the constrained encoder on a batch of gene-expression profiles.

        Math:
            h       = tanh( x @ W_constrained.T + b )
            mu      = fc_mean(h)
            log_var = fc_log_var(h)

        Parameters
        ----------
        x : torch.Tensor, shape (batch, n_genes)

        Returns
        -------
        z_mean : torch.Tensor, shape (batch, latent_dim)
        z_log_var : torch.Tensor, shape (batch, latent_dim)
        h : torch.Tensor, shape (batch, n_pathways)
            Pathway activations - returned so loss() can regularise them.
        """
        raise NotImplementedError

    def reparameterise(self, mu: torch.Tensor, log_var: torch.Tensor) -> torch.Tensor:
        """
        Sample a latent vector z in a differentiable way.

        The reparameterisation trick replaces the non-differentiable sample
        z ~ N(mu, sigma^2) with:

            z = mu + sigma * eps,   eps ~ N(0, I)

        Gradients flow through mu and sigma; eps is just random noise.

        Note: clamp log_var to a finite range before calling exp() to avoid
        numerical overflow.  Scale sigma by a small constant (< 1) to keep
        the KL term manageable at the start of training.

        Parameters
        ----------
        mu : torch.Tensor, shape (batch, latent_dim)
        log_var : torch.Tensor, shape (batch, latent_dim)

        Returns
        -------
        z : torch.Tensor, shape (batch, latent_dim)
        """
        # sigma: standard deviation for sampling, derived from log_var
        # eps: random noise, same shape as sigma
        raise NotImplementedError

    def decode(self, z: torch.Tensor) -> torch.Tensor:
        """
        Map a latent vector back to gene-expression space.

        Math:
            h_prime = tanh( dec_latent(z) )
            x_hat   = dec_out(h_prime)       <- no activation

        No activation on the output - the reconstruction loss (MSE) works
        best with unbounded real values.

        Parameters
        ----------
        z : torch.Tensor, shape (batch, latent_dim)

        Returns
        -------
        x_hat : torch.Tensor, shape (batch, n_genes)
        """
        # h_prime: intermediate pathway representation in the decoder
        raise NotImplementedError

    def forward(self, x: torch.Tensor):
        """
        Full forward pass: encode -> reparameterise -> decode.

        Parameters
        ----------
        x : torch.Tensor, shape (batch, n_genes)

        Returns
        -------
        recon   : torch.Tensor, shape (batch, n_genes)   - reconstructed input
        mu      : torch.Tensor, shape (batch, latent_dim)
        log_var : torch.Tensor, shape (batch, latent_dim)
        h       : torch.Tensor, shape (batch, n_pathways) - pathway activations
        """
        # z: latent sample drawn via reparameterise
        raise NotImplementedError

    # -------------------------------------------------------------------

    def loss(  # pylint: disable=too-many-arguments,too-many-positional-arguments
        self,
        x: torch.Tensor,
        recon: torch.Tensor,
        mu: torch.Tensor,
        log_var: torch.Tensor,
        h: torch.Tensor,
    ) -> torch.Tensor:
        """
        VAE loss with three terms:

            L = L_recon + L_KL + L_L2

        Reconstruction loss - mean squared error scaled by n_genes to keep
        the magnitude stable across datasets of different widths:

            L_recon = MSE(x, x_hat) * n_genes

        KL divergence - closed-form KL between the approximate posterior
        q(z|x) = N(mu, sigma^2) and the standard Gaussian prior p(z) = N(0, I):

            L_KL = -0.5 * mean( 1 + log_var - mu^2 - exp(log_var) )

        L2 regularisation - penalises large pathway activations to keep
        the biological representation sparse and interpretable:

            L_L2 = lambda * mean( h^2 )

        Parameters
        ----------
        x     : torch.Tensor, shape (batch, n_genes)    - original input
        recon : torch.Tensor, shape (batch, n_genes)    - reconstruction
        mu    : torch.Tensor, shape (batch, latent_dim)
        log_var : torch.Tensor, shape (batch, latent_dim)
        h     : torch.Tensor, shape (batch, n_pathways) - pathway activations

        Returns
        -------
        loss : torch.Tensor, scalar
        """
        # recon_loss: reconstruction term
        # kl_loss: KL divergence term
        # l2_loss: L2 regularisation on pathway activations
        raise NotImplementedError
