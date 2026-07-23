"""
Post-hoc interpretation read-outs on top of a trained InformedVAE.

Two functions live here, both operating on the encoder path only (so they
work regardless of which decoder the model was built with):

    bayes_factor_da       signed differential-activation Bayes factor per
                          pathway between two groups of cells
    integrated_gradients  per-gene attribution for a single pathway's
                          activation, via Integrated Gradients (Sundararajan
                          et al., 2017)

Both functions compare the ``h`` output of ``Encoder.forward`` (the raw
pathway activations), not ``z`` or ``mu``. Issue #14 deliberately kept ``h``
un-concatenated with the covariate; this is the cleanest interpretable
signal because it carries no sampling noise from ``reparameterise`` and no
covariate confound.
"""

from __future__ import annotations

import pandas as pd
import torch

from pyvae.models import InformedVAE


def bayes_factor_da(
    model: InformedVAE,
    x_a: torch.Tensor,
    x_b: torch.Tensor,
    cov_a: torch.Tensor | None = None,
    cov_b: torch.Tensor | None = None,
    n_pairs: int = 2000,
    pathway_names: list[str] | None = None,
    seed: int | None = None,
) -> pd.DataFrame:
    """Signed differential-activation Bayes factor per pathway, group A vs B.

    Encodes ``x_a`` and ``x_b`` (with ``cov_a`` / ``cov_b`` if the model is
    conditional) to obtain their raw pathway activations ``h_a`` and ``h_b``,
    Monte-Carlo samples ``n_pairs`` random ``(a, b)`` pairs per pathway, and
    estimates ``p = P(h_a > h_b)`` with ties split 0.5 / 0.5 (so a pathway
    with identical activation in both groups gets ``BF ~= 0`` rather than
    ``+/- inf``). ``p`` is clipped to ``[eps, 1 - eps]`` before the log-odds
    so a single lopsided sample can't produce an infinite BF.

    The sign of the returned BF is meaningful: positive means the pathway
    sits higher in group A than in group B.

    Parameters
    ----------
    model : trained InformedVAE.
    x_a : (n_a, n_genes) log1p-normalized expression for group A.
    x_b : (n_b, n_genes) log1p-normalized expression for group B.
    cov_a, cov_b : optional one-hot covariates for the two groups. Required
        iff the model was built with ``n_cov > 0``.
    n_pairs : number of Monte-Carlo pairs to draw per pathway. Larger is
        less noisy; a few thousand is usually sufficient.
    pathway_names : optional list of length ``n_pathways``. Used as the
        DataFrame index; if not provided, integer indices ``0..n_pathways-1``
        are used.
    seed : optional RNG seed for the pair sampling, for reproducibility. When
        ``None`` (default), the current global torch RNG state is used.

    Returns
    -------
    DataFrame indexed by pathway, columns ``["bf", "p"]``, sorted by
    ``|bf|`` descending.
    """
    eps = 1e-4

    model.eval()
    with torch.no_grad():
        _, _, h_a = model.encode(x_a, cov_a)
        _, _, h_b = model.encode(x_b, cov_b)

    n_a, n_pathways = h_a.shape
    n_b, _ = h_b.shape

    if seed is not None:
        generator = torch.Generator(device=h_a.device)
        generator.manual_seed(seed)
    else:
        generator = None

    idx_a = torch.randint(0, n_a, (n_pairs,), generator=generator, device=h_a.device)
    idx_b = torch.randint(0, n_b, (n_pairs,), generator=generator, device=h_a.device)

    # Per-pathway comparisons over the sampled pairs.
    # Shape: (n_pairs, n_pathways)
    a_sampled = h_a[idx_a]
    b_sampled = h_b[idx_b]

    # Ties get 0.5, strict inequality gets 1.0 or 0.0.
    gt = (a_sampled > b_sampled).float()
    eq = (a_sampled == b_sampled).float()
    p = (gt + 0.5 * eq).mean(dim=0)  # shape (n_pathways,)

    # Clip to avoid log(0) / log(inf) from a fully-lopsided sample.
    p_clipped = p.clamp(min=eps, max=1.0 - eps)
    bf = torch.log(p_clipped / (1.0 - p_clipped))

    if pathway_names is None:
        index: list[str] | pd.Index = pd.RangeIndex(n_pathways, name="pathway")
    else:
        if len(pathway_names) != n_pathways:
            raise ValueError(
                f"pathway_names has length {len(pathway_names)}, "
                f"expected n_pathways={n_pathways}"
            )
        index = pd.Index(pathway_names, name="pathway")

    df = pd.DataFrame(
        {"bf": bf.cpu().numpy(), "p": p.cpu().numpy()},
        index=index,
    )
    return df.reindex(df["bf"].abs().sort_values(ascending=False).index)


def integrated_gradients(
    model: InformedVAE,
    x: torch.Tensor,
    pathway_idx: int,
    cov: torch.Tensor | None = None,
    baseline: torch.Tensor | None = None,
    steps: int = 50,
) -> torch.Tensor:
    """Attribute pathway ``pathway_idx``'s activation to each input gene.

    Integrated Gradients (Sundararajan et al., 2017) attribution for the
    activation of a single pathway. Interpolates ``x`` linearly from a
    baseline (default: ``zeros_like(x)``, an "all genes off" reference cell)
    to the real cell in ``steps`` equal points, accumulates the gradient of
    ``h[:, pathway_idx]`` with respect to the input at each interpolation
    point, then scales the averaged gradient by ``(x - baseline)``.

    Attributes the encoder output ``h``, not ``z`` or ``mu``. The decoder is
    not touched, so this works identically for any decoder variant.

    Parameters
    ----------
    model : trained InformedVAE.
    x : (batch, n_genes) log1p-normalized expression of the real cells.
    pathway_idx : integer index of the pathway to attribute.
    cov : optional one-hot covariates. Required iff the model was built with
        ``n_cov > 0``.
    baseline : (batch, n_genes) reference input. Defaults to
        ``torch.zeros_like(x)``.
    steps : number of Riemann-sum points on the interpolation path.
        The classic IG default (50) is a good balance of accuracy and cost.

    Returns
    -------
    attribution : (batch, n_genes) per-gene attribution. Positive means that
        gene pushed the pathway activation up relative to the baseline.
    """
    if baseline is None:
        baseline = torch.zeros_like(x)

    # Interpolation coefficients: 1/steps, 2/steps, ..., 1.0
    # (Trapezoidal or midpoint would be marginally more accurate; simple
    # right-Riemann matches the reference IG implementation.)
    alphas = torch.linspace(1.0 / steps, 1.0, steps, device=x.device)

    accumulated_gradient = torch.zeros_like(x)

    for alpha in alphas:
        interpolated = baseline + alpha * (x - baseline)
        interpolated = interpolated.detach().clone().requires_grad_(True)

        _mu, _log_var, h = model.encode(interpolated, cov)
        # Scalar objective: sum the pathway activation over the batch.
        # This lets us call .backward() once and get per-sample gradients.
        pathway_activation = h[:, pathway_idx].sum()

        grad = torch.autograd.grad(pathway_activation, interpolated)[0]
        accumulated_gradient = accumulated_gradient + grad

    averaged_gradient = accumulated_gradient / steps
    return (x - baseline) * averaged_gradient
