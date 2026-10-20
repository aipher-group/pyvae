"""InformedVAE — the top-level pathway-informed variational autoencoder.

Phase 1c added five keyword-only options that are threaded through to the
encoder and (for the informed_decoder flag) the decoder:

- ``init`` in {"xavier", "fan_in"} — weight initialisation for masked layers.
- ``normalize`` in {"none", "batch", "layer"} — pre-activation standardisation
  in the encoder's InformedLinear.
- ``informed_decoder`` — when True, the decoder's ``dec_out`` layer is masked
  by ``adj`` too, so each pathway's decoder output only affects its member
  genes.
- ``informed_encoder`` — when False, the encoder's mask is dropped and every
  unit reads every gene. Defaults to True, which is what the encoder has
  always done. It exists so the encoder mask can be ablated the way
  ``informed_decoder`` ablates the decoder's, with everything else held fixed.
- ``nonneg_encoder`` — when True, the encoder's InformedLinear uses softplus
  reparametrisation so every live weight stays positive. Deliberately named
  ``nonneg_encoder`` (not ``nonneg``) to make explicit that this constraint
  applies to the encoder side only; ``nonneg_decoder`` is a separate flag.
- ``standardize_input`` — when True, the encoder z-scores its input per gene
  before the masked matmul, so no single loud gene can dominate its unit.

Every option defaults to the pre-existing behaviour; the golden regression
loss value still reproduces bit-for-bit.
"""

from __future__ import annotations

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
        *,
        init: str = "xavier",
        normalize: str = "none",
        informed_decoder: bool = False,
        informed_encoder: bool = True,
        nonneg_encoder: bool = False,
        standardize_input: bool = False,
        cov_mode: str = "gene",
        cov_in_encoder: bool = True,
        nonneg_decoder: bool = False,
    ):
        super().__init__()
        torch.manual_seed(seed)

        if cov_mode not in ("gene", "pathway"):
            raise ValueError(
                f'cov_mode must be "gene" or "pathway"; got {cov_mode!r}'
            )
        if nonneg_decoder and not informed_decoder:
            raise ValueError(
                "nonneg_decoder=True requires informed_decoder=True: a dense "
                "decoder has no pathway units to orient, so constraining its "
                "signs would restrict the model without making anything "
                "interpretable"
            )
        if nonneg_encoder and not informed_encoder:
            raise ValueError(
                "nonneg_encoder=True requires informed_encoder=True: with no "
                "encoder mask every unit reads every gene, so positive weights "
                "would constrain the whole first layer without making any unit "
                "a pathway. Set both to False to ablate the encoder mask."
            )
        if cov_mode == "pathway" and n_cov == 0:
            raise ValueError(
                'cov_mode="pathway" requires n_cov > 0; with no covariate there '
                "is nothing to route through the pathway layer"
            )

        n_genes, n_pathways = adj.shape

        self.n_genes = n_genes
        self.n_pathways = n_pathways
        self.latent_dim = latent_dim if latent_dim is not None else n_pathways // 2
        self.l2_lambda = l2_lambda
        self.beta = beta
        self.likelihood_kind = likelihood
        self.n_cov = n_cov

        # Phase 1c options: store for introspection and downstream code
        # (e.g. read-outs that want to know whether nonneg is on).
        self.init_kind = init
        self.normalize_kind = normalize
        self.informed_decoder = informed_decoder
        self.informed_encoder = informed_encoder
        self.nonneg_encoder = nonneg_encoder
        self.standardize_input_enabled = standardize_input
        self.cov_mode = cov_mode
        self.cov_in_encoder = cov_in_encoder
        self.nonneg_decoder = nonneg_decoder

        self.encoder = Encoder(
            adj=adj,
            latent_dim=self.latent_dim,
            n_cov=n_cov,
            init=init,
            normalize=normalize,
            nonneg=nonneg_encoder,
            standardize_input=standardize_input,
            use_cov=cov_in_encoder,
            use_mask=informed_encoder,
        )

        # Pass adj to the decoder only when informed_decoder is on.
        #
        # `nonneg_encoder` and `nonneg_decoder` are separate flags and neither
        # implies the other. An earlier version of this comment claimed a
        # non-negative decoder "cannot lower a gene", which is why the option
        # was never wired; it is false. The pathway layer is a tanh and may be
        # negative, so `dec_out` with positive weights still lowers a gene when
        # its unit falls -- the direction moves from the weight to the unit's
        # activity, which is exactly what gives a pathway's sign a meaning.
        # Without it, measured on trained fits, `dec_out` is ~50% negative and
        # the per-unit fraction of positive weights has median 0.500, so a unit
        # rising raises half its genes and lowers the other half.
        #
        # `standardize_input` still does NOT reach the decoder: a BatchNorm on
        # latent-derived activations would fight the reconstruction loss.
        dec_adj = adj if informed_decoder else None

        if likelihood == "gaussian":
            self.decoder = DenseDecoder(
                latent_dim=self.latent_dim,
                n_pathways=self.n_pathways,
                n_genes=self.n_genes,
                n_cov=n_cov,
                adj=dec_adj,
                init=init,
                nonneg=nonneg_decoder,
                cov_mode=cov_mode,
            )
            self.likelihood = GaussianLikelihood()
        elif likelihood == "nb":
            self.decoder = CountDecoder(
                latent_dim=self.latent_dim,
                n_pathways=self.n_pathways,
                n_genes=self.n_genes,
                n_cov=n_cov,
                adj=dec_adj,
                init=init,
                nonneg=nonneg_decoder,
                cov_mode=cov_mode,
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
    def pathway_contrast(
        self,
        cov_a: torch.Tensor,
        cov_b: torch.Tensor,
    ) -> torch.Tensor:
        """The `a -> b` covariate effect, as one signed number per pathway.

        Returns ``cov_path.weight @ (cov_b - cov_a)``, shape ``(n_pathways,)``:
        the shift the covariate imposes on each pathway unit's pre-activation.
        It is cell-independent and exact -- no cells, no sampling, no
        counterfactual -- and positive means the pathway goes up.

        "Up" means what it says only when ``nonneg_decoder=True``. Otherwise
        ``dec_out`` has free signs, each unit's polarity is arbitrary, and this
        vector ranks the right answer no better than chance: measured on Kang,
        the known target sits at rank 1,349 of 1,615 by this quantity without the
        constraint and at 22 with it. The constraint is not enforced here,
        because an unconstrained model's contrast is still the correct
        *magnitude* per unit, but a caller ranking units by sign needs it.

        Requires ``cov_mode="pathway"`` and ``cov_in_encoder=False``. With the
        encoder route open the covariate also reaches the pathway layer through
        ``z``, so this term is one of two additive routes and is not the
        contrast; returning it as though it were is a mistake with measured
        consequences -- on a 125k-cell fit it ranked the known target 1,246th of
        1,615 while the gene-level signal looked correct. Use
        ``pyvae.interpret.unit_contrast`` for that case, which measures both
        routes from the forward pass.

        Parameters
        ----------
        cov_a, cov_b : (n_cov,) or (1, n_cov) or (batch, n_cov)
            The two covariate settings. A batch is reduced by its mean, which is
            exact because the map is linear.

        Returns
        -------
        torch.Tensor
            Shape ``(n_pathways,)``, on the model's device.

        Raises
        ------
        ValueError
            If the model was not built for this read-out, naming the flag that
            is wrong rather than failing on a missing attribute.
        """
        if self.cov_mode != "pathway":
            raise ValueError(
                'pathway_contrast requires cov_mode="pathway"; this model has '
                f'cov_mode="{self.cov_mode}", whose covariate term is dense over '
                "genes and never passes through the pathway mask. Rebuild with "
                'cov_mode="pathway", or use pyvae.interpret.unit_contrast to '
                "measure what reaches the pathway layer in this model."
            )
        if self.cov_in_encoder:
            raise ValueError(
                "pathway_contrast requires cov_in_encoder=False; with the "
                "encoder's covariate route open, cov_path carries only part of "
                "the conditional effect and the rest arrives through z. Rebuild "
                "with cov_in_encoder=False, or use pyvae.interpret.unit_contrast, "
                "which measures both routes."
            )

        w = self.decoder.cov_path.weight  # (n_pathways, n_cov)
        a = torch.as_tensor(cov_a, dtype=w.dtype, device=w.device).reshape(-1, self.n_cov)
        b = torch.as_tensor(cov_b, dtype=w.dtype, device=w.device).reshape(-1, self.n_cov)
        return w @ (b.mean(0) - a.mean(0))

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

        What holding ``mu`` fixed implies, with ``cov_mode="gene"``: the masked
        pathway layer is then identical on both sides and cancels exactly, so the
        predicted shift is precisely ``cov_decoder[:, to] - cov_decoder[:, from]``
        -- one dense weight difference that never touches the annotation.
        Measured against this function's own output, the residual is 5.7e-07.
        That is the intended behaviour of a counterfactual at fixed latent state,
        and it is also why a permuted gene->pathway map leaves the result
        unchanged. With ``cov_mode="pathway"`` the covariate term sits inside the
        pathway layer instead, so the shift does pass through the mask.

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
