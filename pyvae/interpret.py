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
    """Signed logit(Mann-Whitney AUC) per pathway between groups A and B.

    Despite the name, this function does not compute a Bayes factor in the
    sense of expiMap or scVI. It computes ``logit(P(h_a > h_b))`` where the
    probability is estimated by Monte-Carlo sampling of ``n_pairs`` random
    ``(a, b)`` pairs from the two groups' encoder outputs, with ties split
    0.5. That probability, with ties split, is exactly the Mann-Whitney AUC
    statistic — which ``scipy.stats.mannwhitneyu`` computes in closed form
    without any sampling noise. The Monte-Carlo pair sampling here is a
    Bayes-factor-shaped ritual that costs precision without buying anything.

    Nothing in this function integrates over a posterior. ``h`` is the raw
    output of the encoder's masked linear + tanh — deterministic given ``x``
    and (if present) ``cov``. A genuine differential-activation Bayes factor
    would need ``n_samples`` draws of ``z`` from ``q(z|x)`` per cell, then
    a log-ratio of posterior probabilities. See ``differential_expression``
    for the decoder-side equivalent that does that properly.

    What the sign of the returned value reports is the direction of a
    latent unit, not the direction of the underlying pathway's biology.
    If the encoder learned an inverted representation of a pathway (which
    ``pathway_unit_fidelity`` will show by returning ``sign = -1`` for
    that pathway), a positive value here means the *unit* is up-shifted
    in group A — which corresponds to the pathway being *down-shifted*.
    Interpretation therefore requires cross-referencing this function's
    output with fidelity signs, or constraining the encoder so signs
    are guaranteed positive (``nonneg_encoder=True`` on ``InformedVAE``).

    Kept named ``bayes_factor_da`` to preserve caller compatibility; the
    function's behaviour is unchanged. The docstring is what changed, so
    downstream text describing "the Bayes factor of the interferon pathway"
    can be rewritten with accurate language.

    Parameters
    ----------
    model : trained InformedVAE.
    x_a : (n_a, n_genes) log1p-normalized expression for group A.
    x_b : (n_b, n_genes) log1p-normalized expression for group B.
    cov_a, cov_b : optional one-hot covariates for the two groups. Required
        iff the model was built with ``n_cov > 0``.
    n_pairs : number of Monte-Carlo pairs to draw per pathway. Larger is
        less noisy; a few thousand is usually sufficient. Set to a value
        much smaller than ``n_a * n_b`` — otherwise you're doing a slow,
        noisy approximation of the exact Mann-Whitney AUC.
    pathway_names : optional list of length ``n_pathways``. Used as the
        DataFrame index; if not provided, integer indices ``0..n_pathways-1``
        are used.
    seed : optional RNG seed for the pair sampling, for reproducibility.
        When ``None`` (default), the current global torch RNG state is used.

    Returns
    -------
    DataFrame indexed by pathway, columns ``["bf", "p"]``, sorted by
    ``|bf|`` descending. Column ``bf`` is ``logit(p)`` clipped to
    ``[eps, 1-eps]`` before taking logs so a fully-lopsided sample cannot
    produce ``+/- inf``. Despite the name ``bf``, this is not a Bayes
    factor; see the docstring above.
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

def pathway_unit_fidelity(
    model,
    x,
    adj,
    *,
    cov=None,
    min_genes=10,
    batch_size=1024,
    eps=1e-8,
):
    """Correlate each pathway unit's activation against its member genes.

    For each pathway j, compute the Pearson correlation between the encoder's
    pathway-layer activation h[:, j] and the mean z-scored expression of the
    genes annotated to pathway j in the adjacency matrix. A unit with |corr|
    near 1 tracks its pathway's activity as a whole; a unit near 0 does not
    track it at all; a unit with negative corr encodes the pathway upside-down.

    This is a diagnostic instrument, not a metric to be optimized. Its purpose
    is to reveal whether InformedLinear's gene-membership constraint alone is
    enough to make a unit represent its pathway, or whether the encoder's
    freedom in weight signs and magnitudes lets a unit drift into being a
    detector for one loud member gene instead.

    Parameters
    ----------
    model : InformedVAE
        A trained model. Must have ``.encode(x[, cov]) -> (mu, logvar, h)`` or
        expose ``h`` through the encoder in some other documented way. The
        function calls ``model.eval()`` and runs inside ``torch.no_grad()``.
    x : torch.Tensor or numpy.ndarray or pandas.DataFrame
        Expression matrix of shape (n_cells, n_genes). Must have the same
        gene axis as ``adj`` (i.e. same order of columns as adj's rows).
    adj : torch.Tensor or numpy.ndarray or pandas.DataFrame
        Binary gene-to-pathway adjacency of shape (n_genes, n_pathways).
        If a DataFrame is passed, its columns are used as pathway names in
        the returned DataFrame index. If not, pathways are named "pathway_j".
    cov : torch.Tensor or numpy.ndarray, optional
        Covariate matrix of shape (n_cells, n_cov). Required if the model was
        trained with ``n_cov > 0``. Raises ValueError if the model needs a
        covariate and none is given.
    min_genes : int, default 10
        Pathways with fewer than this many annotated genes are skipped.
        Their ``corr`` field will be NaN. A correlation on very few genes
        (e.g. 3) is noisy enough to mislead more than inform.
    batch_size : int, default 1024
        Number of cells to encode per batch. Purely a memory-management
        choice; does not affect the result.
    eps : float, default 1e-8
        Small constant added to standard deviations before dividing, to
        avoid divide-by-zero when a gene has zero variance in the sample.

    Returns
    -------
    pandas.DataFrame
        One row per pathway, indexed by pathway name. Columns:

        - ``n_genes`` : number of annotated genes for this pathway
        - ``corr`` : Pearson correlation between h[:, j] and the mean z-scored
          expression of the pathway's member genes. NaN if n_genes < min_genes
          or if the unit had zero variance across cells.
        - ``abs_corr`` : absolute value of ``corr``
        - ``sign`` : +1 if corr > 0, -1 if corr < 0, 0 if corr is NaN or 0

    Notes
    -----
    The z-scoring is per gene across cells, not per cell across genes. This
    matches the standard "expression signature" convention: a gene's activity
    is measured relative to its own distribution across the sample.

    The mean-z aggregation across a pathway's member genes gives all members
    equal weight. This is by design: the failure mode we care about is a unit
    dominated by one loud gene, and comparing that unit to an unweighted mean
    of all members is exactly how we detect the domination.

    Comparing this function's output across two models trained with different
    encoder constraints (e.g. baseline vs. nonneg+standardize) is the primary
    use. A model where fidelity improves under constraints is a model where
    the constraints removed a real freedom that was hurting representation.

    Examples
    --------
    >>> fid = pathway_unit_fidelity(model, x, adj_df)
    >>> fid.sort_values("abs_corr", ascending=False).head()
    >>> print(f"median |corr|: {fid['abs_corr'].median():.3f}")
    >>> print(f"inverted fraction: {(fid['sign'] < 0).mean():.3f}")
    """
    import numpy as np
    import pandas as pd
    import torch

    # --- Coerce inputs ---
    if isinstance(x, pd.DataFrame):
        x_arr = x.values
    elif isinstance(x, np.ndarray):
        x_arr = x
    elif isinstance(x, torch.Tensor):
        x_arr = x.detach().cpu().numpy()
    else:
        raise TypeError(f"x must be Tensor, ndarray, or DataFrame; got {type(x).__name__}")
    x_arr = np.asarray(x_arr, dtype=np.float32)

    if isinstance(adj, pd.DataFrame):
        pathway_names = list(adj.columns)
        adj_arr = adj.values
    elif isinstance(adj, np.ndarray):
        adj_arr = adj
        pathway_names = [f"pathway_{j}" for j in range(adj.shape[1])]
    elif isinstance(adj, torch.Tensor):
        adj_arr = adj.detach().cpu().numpy()
        pathway_names = [f"pathway_{j}" for j in range(adj.shape[1])]
    else:
        raise TypeError(f"adj must be Tensor, ndarray, or DataFrame; got {type(adj).__name__}")
    adj_arr = np.asarray(adj_arr, dtype=np.float32)

    # --- Shape checks ---
    n_cells, n_genes = x_arr.shape
    n_genes_adj, n_pathways = adj_arr.shape
    if n_genes != n_genes_adj:
        raise ValueError(
            f"Gene axis mismatch: x has {n_genes} genes, adj has {n_genes_adj}."
        )

    # --- Covariate handling ---
    model_n_cov = getattr(model, "n_cov", 0)
    if model_n_cov > 0 and cov is None:
        raise ValueError(
            f"Model was trained with n_cov={model_n_cov} but no cov was provided."
        )
    if model_n_cov == 0 and cov is not None:
        raise ValueError(
            "Model has n_cov=0 but a cov was provided. Pass cov=None."
        )
    if cov is not None:
        if isinstance(cov, pd.DataFrame):
            cov_arr = cov.values
        elif isinstance(cov, np.ndarray):
            cov_arr = cov
        elif isinstance(cov, torch.Tensor):
            cov_arr = cov.detach().cpu().numpy()
        else:
            raise TypeError(f"cov must be Tensor, ndarray, or DataFrame; got {type(cov).__name__}")
        cov_arr = np.asarray(cov_arr, dtype=np.float32)
        if cov_arr.shape[0] != n_cells:
            raise ValueError(
                f"cov has {cov_arr.shape[0]} rows but x has {n_cells} cells."
            )

    # --- Encode to get h ---
    device = next(model.parameters()).device
    was_training = model.training
    model.eval()

    h_batches = []
    try:
        with torch.no_grad():
            for i in range(0, n_cells, batch_size):
                x_b = torch.from_numpy(x_arr[i : i + batch_size]).to(device)
                if cov is not None:
                    c_b = torch.from_numpy(cov_arr[i : i + batch_size]).to(device)
                    _, _, h_b = model.encode(x_b, c_b)
                else:
                    _, _, h_b = model.encode(x_b)
                h_batches.append(h_b.detach().cpu().numpy())
    finally:
        if was_training:
            model.train()

    h_arr = np.concatenate(h_batches, axis=0)
    assert h_arr.shape == (n_cells, n_pathways), (
        f"Expected h shape ({n_cells}, {n_pathways}), got {h_arr.shape}. "
        "Does model.encode return (mu, logvar, h) as expected?"
    )

    # --- Z-score expression per gene across cells ---
    x_mean = x_arr.mean(axis=0, keepdims=True)
    x_std = x_arr.std(axis=0, keepdims=True)
    x_z = (x_arr - x_mean) / (x_std + eps)  # shape (n_cells, n_genes)

    # --- Per-pathway: correlate h[:, j] with mean z of member genes ---
    n_genes_per_pathway = adj_arr.sum(axis=0).astype(int)  # shape (n_pathways,)
    corr = np.full(n_pathways, np.nan, dtype=np.float64)

    for j in range(n_pathways):
        n_j = n_genes_per_pathway[j]
        if n_j < min_genes:
            continue  # corr stays NaN

        # Mean z-scored expression across the j-th pathway's members
        member_mask = adj_arr[:, j].astype(bool)   # shape (n_genes,)
        mean_z_j = x_z[:, member_mask].mean(axis=1)  # shape (n_cells,)

        h_j = h_arr[:, j]  # shape (n_cells,)

        # Pearson correlation; guard against zero-variance columns
        h_std = h_j.std()
        mz_std = mean_z_j.std()
        if h_std < eps or mz_std < eps:
            continue  # corr stays NaN — unit or member-mean is degenerate

        h_c = h_j - h_j.mean()
        mz_c = mean_z_j - mean_z_j.mean()
        corr[j] = float(np.mean(h_c * mz_c) / (h_std * mz_std))

    abs_corr = np.abs(corr)
    sign = np.where(np.isnan(corr), 0, np.sign(corr)).astype(int)

    return pd.DataFrame(
        {
            "n_genes": n_genes_per_pathway,
            "corr": corr,
            "abs_corr": abs_corr,
            "sign": sign,
        },
        index=pd.Index(pathway_names, name="pathway"),
    )

