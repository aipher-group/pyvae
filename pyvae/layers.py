"""InformedLinear with encoder-constraint options.

The base layer masks a dense linear so unit j reads only its pathway's member
genes. This file extends it with five independent options that constrain
*how* a unit uses those genes, not just which:

- `init` : where to start the weights (per-unit fan-in-aware, or uniform xavier)
- `normalize` : whether to standardise the pre-activation each batch
- `nonneg` : whether every weight is forced positive (removes sign freedom)
- `standardize_input` : whether to z-score the input first (removes scale freedom)
- `mask_bias` : whether to zero the bias for pathways with no members

Every option defaults to the pre-existing behaviour, so the golden regression
test still passes bit-for-bit.
"""
from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


_ALLOWED_ACTIVATIONS = ("tanh", "relu", "linear")
_ALLOWED_INITS = ("xavier", "fan_in")
_ALLOWED_NORMALIZE = ("none", "batch", "layer")


class InformedLinear(nn.Module):
    """Masked linear layer with optional constraints on how weights behave.

    In the base configuration (all defaults), this reproduces the original
    behaviour: xavier-initialised dense weights multiplied element-wise by
    a binary mask so unit j reads only pathway j's member genes, plus a
    bias masked to zero for pathways with no members.

    The five keyword-only options change how a unit is *allowed* to use its
    inputs, not what it may read. They matter because the reconstruction loss
    never tells unit j to resemble pathway j — it only asks the model to
    reconstruct. Without further constraint, a unit can encode its pathway
    upside-down (negative weights) or become a detector for whichever member
    gene has the largest variance. Both failure modes are diagnosable with
    ``pathway_unit_fidelity`` and correctable by combinations of the options
    below.

    Parameters
    ----------
    adj : torch.Tensor
        Binary adjacency of shape (n_genes, n_pathways). ``adj[i, j] == 1``
        means gene i is a member of pathway j. Stored as a non-trainable
        buffer, transposed to (n_pathways, n_genes) for use in the forward.
    activation : {"tanh", "relu", "linear"}, default "tanh"
        Applied after the masked linear and (if enabled) normalisation.
        "linear" leaves the pre-activation untouched.
    init : {"xavier", "fan_in"}, default "xavier"
        Weight initialisation. "xavier" is torch's ``xavier_uniform_``, which
        derives its bound from the full ``(out_f, in_f)`` shape and hands
        every unit the same scale regardless of how many genes it actually
        reads. "fan_in" uses a per-unit bound ``sqrt(3 / n_genes_in_pathway_j)``,
        so each unit's initial pre-activation has variance 1 regardless of
        its pathway size. Units with no live inputs get bound 0 (weights
        stay at exactly 0), avoiding the ``sqrt(3/0) = inf`` trap.
    normalize : {"none", "batch", "layer"}, default "none"
        Standardise the pre-activation each batch. "batch" is a non-affine
        ``BatchNorm1d(out_f, affine=False)``; "layer" is a non-affine
        ``LayerNorm(out_f, elementwise_affine=False)``. Both subtract a mean,
        which means the bias term becomes inert when normalize != "none".
        Not silently: pick your control over scale (init) or your control
        over scale for the whole run (normalize) — they are not interchangeable.
    mask_bias : bool, default True
        When True, the bias is multiplied by ``bias_mask`` (1 for pathways
        with members, 0 for pathways with none), so an orphan pathway
        stays at exactly zero. When False, every pathway has a free bias.
    nonneg : bool, default False
        When True, weights are stored as an unconstrained parameter and
        passed through ``softplus`` before the mask multiply, so every live
        weight is strictly positive. This removes sign freedom: a unit
        then rises monotonically with every one of its member genes,
        making a positive Bayes factor mean the pathway is upregulated
        rather than merely reflecting some inverted encoder weights.
        Initialisation is arranged so ``softplus(raw)`` reproduces the
        magnitude the chosen ``init`` requested, rather than collapsing
        everything onto ``softplus(0) = 0.693``.
    standardize_input : bool, default False
        When True, apply a non-affine ``BatchNorm1d(in_f)`` to ``x`` before
        the masked matmul. This removes scale freedom on the input side:
        one loud gene can no longer swamp its pathway's unit simply by
        virtue of having a large variance.

    Attributes
    ----------
    weight : nn.Parameter
        Shape ``(out_f, in_f)``. When ``nonneg=True``, this is a pre-softplus
        parameter, not the effective weight — use ``effective_weight()``
        instead of reading ``weight`` directly.
    bias : nn.Parameter
        Shape ``(out_f,)``.
    mask : torch.Tensor (buffer)
        Shape ``(out_f, in_f)``. Never modified during training.
    bias_mask : torch.Tensor (buffer)
        Shape ``(out_f,)``.

    Raises
    ------
    ValueError
        If ``activation``, ``init``, or ``normalize`` is not in its allowed
        tuple. The message names both the invalid value and the allowed set.

    Notes
    -----
    ``effective_weight()`` returns the exact ``weight ⊙ mask`` (or
    ``softplus(weight) ⊙ mask`` when ``nonneg=True``) that the forward pass
    uses. Read-outs like ``pathway_unit_fidelity`` and any decoder-side
    interpretation should use it rather than the raw ``weight`` parameter,
    otherwise the interpretation and the model can disagree.
    """

    def __init__(
        self,
        adj: torch.Tensor,
        activation: str = "tanh",
        *,
        init: str = "xavier",
        normalize: str = "none",
        mask_bias: bool = True,
        nonneg: bool = False,
        standardize_input: bool = False,
    ):
        super().__init__()

        if activation not in _ALLOWED_ACTIVATIONS:
            raise ValueError(
                f"activation must be one of {_ALLOWED_ACTIVATIONS}; got {activation!r}"
            )
        if init not in _ALLOWED_INITS:
            raise ValueError(
                f"init must be one of {_ALLOWED_INITS}; got {init!r}"
            )
        if normalize not in _ALLOWED_NORMALIZE:
            raise ValueError(
                f"normalize must be one of {_ALLOWED_NORMALIZE}; got {normalize!r}"
            )

        in_f, out_f = adj.shape
        self.in_f = in_f
        self.out_f = out_f
        self.activation = activation
        self.init_kind = init
        self.normalize_kind = normalize
        self.mask_bias_enabled = mask_bias
        self.nonneg = nonneg
        self.standardize_input_enabled = standardize_input

        self.weight = nn.Parameter(torch.empty(out_f, in_f))
        self.bias = nn.Parameter(torch.zeros(out_f))
        self.register_buffer("mask", adj.T.float())
        self.register_buffer("bias_mask", (adj.sum(dim=0) > 0).float())

        # Initialise weights. If nonneg is on, the parameter we store is a
        # pre-softplus value; we set it so softplus reproduces the intended
        # magnitude rather than collapsing every weight onto softplus(0) = 0.693.
        self._initialise_weights()

        # Optional pre-activation normaliser.
        if normalize == "batch":
            self.pre_norm = nn.BatchNorm1d(out_f, affine=False)
        elif normalize == "layer":
            self.pre_norm = nn.LayerNorm(out_f, elementwise_affine=False)
        else:
            self.pre_norm = None

        # Optional input standardiser.
        if standardize_input:
            self.input_norm = nn.BatchNorm1d(in_f, affine=False)
        else:
            self.input_norm = None

    # ------------------------------------------------------------------
    # Initialisation
    # ------------------------------------------------------------------

    def _initialise_weights(self) -> None:
        """Fill self.weight following ``init_kind``, honouring ``nonneg``."""
        # Target magnitudes first — as if nonneg were off.
        if self.init_kind == "xavier":
            target = torch.empty_like(self.weight)
            nn.init.xavier_uniform_(target)
        elif self.init_kind == "fan_in":
            # Per-unit uniform: sqrt(3 / n_genes_in_pathway_j).
            # Units with no live inputs get bound 0 so their weights stay at 0.
            n_genes_per_pathway = self.mask.sum(dim=1)  # shape (out_f,)
            with torch.no_grad():
                target = torch.empty_like(self.weight)
                for j in range(self.out_f):
                    n_in = n_genes_per_pathway[j].item()
                    if n_in <= 0:
                        target[j].zero_()
                    else:
                        bound = (3.0 / n_in) ** 0.5
                        target[j].uniform_(-bound, bound)
        else:  # pragma: no cover — guarded in __init__
            raise ValueError(f"unknown init: {self.init_kind}")

        with torch.no_grad():
            if self.nonneg:
                # Store log(expm1(|target|)) so that softplus(param) recovers
                # the target magnitude. softplus(log(expm1(x))) = x for x > 0.
                # Use a small floor so log(expm1(0)) doesn't produce -inf.
                magnitude = target.abs().clamp(min=1e-4)
                self.weight.copy_(torch.log(torch.expm1(magnitude)))
            else:
                self.weight.copy_(target)

    # ------------------------------------------------------------------
    # Effective weight — the one the forward actually uses
    # ------------------------------------------------------------------

    def effective_weight(self) -> torch.Tensor:
        """Return the weight tensor as the forward pass uses it.

        Shape ``(out_f, in_f)``. When ``nonneg=True``, this is
        ``softplus(weight) * mask``; otherwise ``weight * mask``. Read-outs
        that need to inspect the encoder's weights should use this method,
        not the raw ``weight`` parameter.
        """
        w = F.softplus(self.weight) if self.nonneg else self.weight
        return w * self.mask

    # ------------------------------------------------------------------
    # Forward
    # ------------------------------------------------------------------

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Optional input standardisation (removes scale freedom on the input side).
        if self.input_norm is not None:
            x = self.input_norm(x)

        # Masked linear. effective_weight() handles nonneg + mask together.
        w = self.effective_weight()

        # Bias — masked or free, depending on mask_bias.
        b = self.bias * self.bias_mask if self.mask_bias_enabled else self.bias

        out = x @ w.T + b

        # Optional pre-activation normalisation.
        # Note: BatchNorm and LayerNorm both subtract a mean here, so the bias
        # term above becomes inert whenever this branch runs. Users choosing
        # normalize != "none" should not also expect the bias to shift anything.
        if self.pre_norm is not None:
            out = self.pre_norm(out)

        # Activation.
        if self.activation == "tanh":
            out = torch.tanh(out)
        elif self.activation == "relu":
            out = torch.relu(out)
        # "linear" — no activation

        return out