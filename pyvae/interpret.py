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
import numpy as np

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

def differential_expression(
    model,
    x_a,
    x_b,
    cov_a=None,
    cov_b=None,
    n_samples: int = 25,
    n_pairs: int = 2000,
    delta: float = 0.25,
    gene_names: list[str] | None = None,
    seed: int | None = None,
    eps: float = 1e-8,
    min_detection: float = 0.0,
) -> "pd.DataFrame":
    """Posterior differential expression per gene, group A vs B, on decoder proportions.

    Follows Boyeau et al. 2019. For each of ``n_samples`` rounds, sample one
    ``z`` per cell from ``q(z|x)`` for both groups, decode to per-cell gene
    proportions, then draw ``n_pairs`` random ``(a, b)`` pairs and record
    per-pair, per-gene ``log2`` ratios of proportions. Aggregate across all
    rounds and pairs to get posterior summaries per gene.

    This is the decoder-side counterpart to ``bayes_factor_da`` and is what
    the latter's name pretends to do — the encoder-side function integrates
    nothing over the posterior because ``h`` is deterministic given ``x``.
    Here every round draws fresh ``z`` samples, so the returned ``bf`` is a
    real Monte-Carlo posterior Bayes factor.

    Because gene expression has an unambiguous polarity (a positive
    ``lfc_mean`` means the gene went up, full stop), pathway rankings built
    on top of this function do not need the "cross-reference the sign against
    unit fidelity" caveat that ``bayes_factor_da`` carries. This is why
    ``pathway_activity`` — the read-out that turns per-gene DE into per-pathway
    rankings — is designed to run on this function's output rather than on
    the encoder's ``h``.

    Requires ``model.likelihood_kind == "nb"``. The Gaussian decoder produces
    raw predicted expression rather than proportions, so the log-ratio
    construction below would be misleading; raises otherwise.

    Parameters
    ----------
    model : InformedVAE
        A trained model with ``likelihood_kind == "nb"``.
    x_a : (n_a, n_genes) array-like
        Log1p-normalised expression for group A. Any dtype accepted by
        ``as_float_tensor`` (torch, numpy, pandas, scipy sparse).
    x_b : (n_b, n_genes) array-like
        Same for group B.
    cov_a, cov_b : (n_a, n_cov), (n_b, n_cov) or None
        One-hot covariates. Required iff the model was built with ``n_cov > 0``.
    n_samples : int, default 25
        Number of rounds. Each round draws fresh ``z`` samples for every cell.
        Higher values reduce posterior noise but scale linearly with cost.
    n_pairs : int, default 2000
        Number of random ``(a, b)`` pairs drawn per round. Total pair-samples
        is ``n_samples * n_pairs``.
    delta : float, default 0.25
        DE threshold on ``|log2fc|``. A gene is called "differentially
        expressed" in a sample iff ``|log2fc| > delta``. ``proba_de`` is the
        fraction of samples where this holds.
    gene_names : list of str or None
        Gene labels for the returned DataFrame's index. If None, integer
        indices 0..n_genes-1 are used.
    seed : int or None
        RNG seed. Both the ``z`` sampling and the pair sampling are seeded
        from it, so identical seeds reproduce results bit-for-bit.
    eps : float, default 1e-8
        Small constant added before dividing proportions and taking logs,
        so a gene with a zero-decoded proportion in one cell doesn't produce
        ``inf`` in the log ratio.
    min_detection : float, default 0.0
        Optional threshold on decoded proportions. A pair contributes to the
        ``detection_rate`` count for a gene iff both cells' decoded proportion
        for that gene exceeds ``min_detection``. A gene where
        ``detection_rate`` is much less than 1 has its ``lfc_mean`` and
        ``proba_de`` computed on very few real pairs — usually a sign to
        exclude it from downstream analysis.

    Returns
    -------
    pandas.DataFrame
        One row per gene, sorted by ``proba_de`` descending then
        ``|lfc_mean|`` descending. Sorting matters because ``proba_de`` is
        a fraction of finitely many draws and saturates at exactly 1.0,
        leaving every tied gene in gene-matrix column order (which is not
        an order). Columns:

        - ``proba_de`` : fraction of samples where ``|log2fc| > delta``.
        - ``bf`` : ``logit(proba_de)``. Clipped so that ``|bf|`` never
          exceeds what the total sample size ``n_samples * n_pairs`` can
          justify — specifically, ``proba_de`` is clipped to
          ``[1/(2*total), 1 - 1/(2*total)]`` before the logit. This
          replaces an arbitrary ``eps`` clip with the bound the data
          actually supports.
        - ``lfc_mean`` : mean ``log2`` fold change across all sampled pairs.
        - ``lfc_std`` : standard deviation of ``log2`` fold change.
        - ``proba_up`` : fraction of samples where ``log2fc > 0``. Together
          with ``lfc_mean`` this tells you whether the gene went up in A vs B.
        - ``detection_rate`` : fraction of samples where both cells in the
          pair had decoded proportion above ``min_detection``.

    Notes
    -----
    Complexity is 2 encoder passes plus ``2 * n_samples`` decoder passes —
    the ``mu, log_var`` for both groups are computed once and reused across
    rounds. Defaults (25 rounds × 2000 pairs) are fast on CPU.

    Cost of using proportions (Boyeau's choice) rather than mean counts:
    the returned ``lfc_mean`` is a log-ratio of transcriptional "budget
    shares" rather than absolute concentrations, so a gene that goes from
    1% to 2% of a cell's budget shows ``lfc_mean = 1``, regardless of
    whether library size changed. This is scale-invariant, which is usually
    what you want; if you specifically need concentration changes, decode
    ``px_scale * library`` explicitly instead.

    Read the ``detection_rate`` column of your top-20 genes before trusting
    them. A gene ranked #1 with ``detection_rate=0.02`` is being ranked on
    a handful of pairs and is more likely a sampling artifact than a real
    finding.
    """
    import numpy as np
    import pandas as pd
    import torch

    from pyvae.components import as_float_tensor

    if model.likelihood_kind != "nb":
        raise ValueError(
            f"differential_expression requires model.likelihood_kind == 'nb', "
            f"got {model.likelihood_kind!r}. The Gaussian decoder produces raw "
            f"predicted expression, not per-cell proportions, so the log-ratio "
            f"construction here would be misleading."
        )

    # --- Coerce inputs to float tensors ---
    x_a_t = as_float_tensor(x_a, name="x_a")
    x_b_t = as_float_tensor(x_b, name="x_b")
    if cov_a is not None:
        cov_a_t = as_float_tensor(cov_a, name="cov_a")
    else:
        cov_a_t = None
    if cov_b is not None:
        cov_b_t = as_float_tensor(cov_b, name="cov_b")
    else:
        cov_b_t = None

    # --- Covariate guards ---
    n_cov = getattr(model, "n_cov", 0)
    if n_cov > 0 and (cov_a_t is None or cov_b_t is None):
        raise ValueError(
            f"model was trained with n_cov={n_cov} but cov_a and/or cov_b are None."
        )
    if n_cov == 0 and (cov_a_t is not None or cov_b_t is not None):
        raise ValueError(
            "model has n_cov=0 but cov_a or cov_b was provided. Pass cov=None "
            "for both, or rebuild the model with n_cov > 0."
        )

    # --- Encode both groups once; sample z fresh per round below ---
    device = next(model.parameters()).device
    x_a_t = x_a_t.to(device)
    x_b_t = x_b_t.to(device)
    if cov_a_t is not None:
        cov_a_t = cov_a_t.to(device)
        cov_b_t = cov_b_t.to(device)

    was_training = model.training
    model.eval()
    try:
        with torch.no_grad():
            mu_a, log_var_a, _ = model.encode(x_a_t, cov_a_t)
            mu_b, log_var_b, _ = model.encode(x_b_t, cov_b_t)
    finally:
        if was_training:
            model.train()

    n_a = mu_a.shape[0]
    n_b = mu_b.shape[0]
    n_genes = x_a_t.shape[1]

    # --- Seeded generators for reproducibility ---
    if seed is not None:
        pair_gen = torch.Generator(device=device)
        pair_gen.manual_seed(seed)
        z_gen = torch.Generator(device=device)
        z_gen.manual_seed(seed + 1)
    else:
        pair_gen = None
        z_gen = None

    # --- Accumulators for the per-gene statistics ---
    # Kept on CPU as numpy arrays to avoid keeping (n_samples * n_pairs, n_genes)
    # in GPU memory (that would be ~50k * 5k * 4 bytes = 1 GB for defaults).
    de_count = np.zeros(n_genes, dtype=np.float64)      # |lfc| > delta
    up_count = np.zeros(n_genes, dtype=np.float64)      # lfc > 0
    lfc_sum = np.zeros(n_genes, dtype=np.float64)       # sum of lfc
    lfc_sq_sum = np.zeros(n_genes, dtype=np.float64)    # sum of lfc^2 (for std)
    det_count = np.zeros(n_genes, dtype=np.float64)     # both cells > min_detection
    total_pairs = n_samples * n_pairs

    sigma_a = torch.exp(0.5 * log_var_a)
    sigma_b = torch.exp(0.5 * log_var_b)

    with torch.no_grad():
        for _ in range(n_samples):
            # Reparameterisation: fresh z per cell per round. This is the
            # posterior integration Boyeau requires.
            eps_a = torch.randn(mu_a.shape, generator=z_gen, device=device)
            eps_b = torch.randn(mu_b.shape, generator=z_gen, device=device)
            z_a = mu_a + sigma_a * eps_a
            z_b = mu_b + sigma_b * eps_b

            # Decode all cells to per-cell proportions.
            px_a = model.decode(z_a, cov_a_t)
            px_b = model.decode(z_b, cov_b_t)

            # Sample n_pairs random (a, b) indices for this round.
            idx_a = torch.randint(0, n_a, (n_pairs,), generator=pair_gen, device=device)
            idx_b = torch.randint(0, n_b, (n_pairs,), generator=pair_gen, device=device)

            px_a_paired = px_a[idx_a]  # (n_pairs, n_genes)
            px_b_paired = px_b[idx_b]

            log2fc = torch.log2((px_a_paired + eps) / (px_b_paired + eps))

            # Move to CPU numpy for accumulation.
            log2fc_np = log2fc.cpu().numpy()
            px_a_np = px_a_paired.cpu().numpy()
            px_b_np = px_b_paired.cpu().numpy()

            de_count += (np.abs(log2fc_np) > delta).sum(axis=0)
            up_count += (log2fc_np > 0).sum(axis=0)
            lfc_sum += log2fc_np.sum(axis=0)
            lfc_sq_sum += (log2fc_np ** 2).sum(axis=0)
            detected = (px_a_np > min_detection) & (px_b_np > min_detection)
            det_count += detected.sum(axis=0)

    # --- Compute per-gene summary statistics ---
    proba_de = de_count / total_pairs
    proba_up = up_count / total_pairs
    lfc_mean = lfc_sum / total_pairs
    lfc_var = lfc_sq_sum / total_pairs - lfc_mean ** 2
    lfc_std = np.sqrt(np.maximum(lfc_var, 0))
    detection_rate = det_count / total_pairs

    # Clip proba_de at what the finite sample can support, per Carlos's spec:
    # 1 / (2 * total) rather than an arbitrary constant. This bounds |bf| at
    # what the sample size can actually justify.
    clip_lo = 1.0 / (2.0 * total_pairs)
    p_clipped = np.clip(proba_de, clip_lo, 1.0 - clip_lo)
    bf = np.log(p_clipped / (1.0 - p_clipped))

    # --- Build the DataFrame ---
    if gene_names is None:
        index = pd.RangeIndex(n_genes, name="gene")
    else:
        if len(gene_names) != n_genes:
            raise ValueError(
                f"gene_names has length {len(gene_names)}, expected n_genes={n_genes}"
            )
        index = pd.Index(gene_names, name="gene")

    df = pd.DataFrame(
        {
            "proba_de": proba_de,
            "bf": bf,
            "lfc_mean": lfc_mean,
            "lfc_std": lfc_std,
            "proba_up": proba_up,
            "detection_rate": detection_rate,
        },
        index=index,
    )

    # Sort by proba_de desc, then |lfc_mean| desc. This matters because
    # proba_de saturates at exactly 1.0 for strongly-DE genes, leaving every
    # tied gene in gene-matrix column order (which is not an order).
    df = df.assign(_abs_lfc=lambda d: d["lfc_mean"].abs())
    df = df.sort_values(
        by=["proba_de", "_abs_lfc"],
        ascending=[False, False],
        kind="mergesort",  # stable, so ties break by prior order (which is _abs_lfc)
    ).drop(columns="_abs_lfc")

    return df

def _benjamini_hochberg(pvalues):
    """Benjamini-Hochberg (1995) FDR-adjusted p-values.

    Standard implementation: sort p-values ascending, adjust each by
    ``p * n / rank``, then enforce monotonicity from the largest rank down
    so a smaller adjusted p never exceeds a larger one. Returns adjusted
    p-values in the original order of the input.

    NaN inputs propagate as NaN in the output — they contribute nothing
    to the count ``n``, so pathways skipped for small size don't inflate
    the adjustment burden on the pathways we actually tested.

    Parameters
    ----------
    pvalues : array-like of float, shape (m,)
        Raw p-values from independent tests. May contain NaN entries
        (these pass through untouched).

    Returns
    -------
    numpy.ndarray, shape (m,), dtype float64
        BH-adjusted p-values, clipped to [0, 1]. NaN positions in the
        input remain NaN in the output.

    Notes
    -----
    Behaviour matches ``statsmodels.stats.multitest.multipletests(
    method='fdr_bh')`` for the all-finite case. We implement it inline
    to avoid a statsmodels dependency for a single-function purpose.
    """
    import numpy as np

    p = np.asarray(pvalues, dtype=np.float64)
    n_total = len(p)
    finite_mask = np.isfinite(p)
    finite_p = p[finite_mask]
    n_finite = len(finite_p)

    if n_finite == 0:
        return p.copy()

    # Sort finite p-values ascending and remember the permutation.
    order = np.argsort(finite_p)
    p_sorted = finite_p[order]

    # Raw BH: p[i] * n / (i+1), 1-indexed rank.
    ranks = np.arange(1, n_finite + 1)
    q_sorted = p_sorted * n_finite / ranks

    # Enforce monotonicity from the top: adjusted[i] = min(adjusted[i:]).
    # Equivalent to running minimum right-to-left.
    q_sorted = np.minimum.accumulate(q_sorted[::-1])[::-1]

    # Clip to [0, 1] — the ratio can exceed 1 before clipping.
    q_sorted = np.clip(q_sorted, 0.0, 1.0)

    # Invert the sort to get q-values back in the original finite order.
    q_finite = np.empty_like(q_sorted)
    q_finite[order] = q_sorted

    # Rebuild the full output with NaN in the skipped positions.
    q = np.full(n_total, np.nan, dtype=np.float64)
    q[finite_mask] = q_finite
    return q


def pathway_activity(
    de,
    adj,
    *,
    statistic: str = "lfc_mean",
    min_genes: int = 5,
    center: bool = True,
) -> "pd.DataFrame":
    """Per-pathway competitive Mann-Whitney ranking from a per-gene DE table.

    For each pathway, tests whether the DE statistic (default ``lfc_mean``)
    of its member genes is systematically different from the DE statistic
    of the non-member genes in the panel. This is the "competitive"
    formulation used by GSEA and camera: it compares members to a
    background of non-members, rather than comparing members to a
    hypothesised null of zero ("self-contained").

    Competitive over self-contained is deliberate. A self-contained test
    is dominated by whichever single member gene moved most — the exact
    "one loud gene wins" failure mode that ``differential_expression``'s
    predecessor suffered from on the encoder side. Testing against the
    rest of the panel asks whether the pathway *as a whole* stands out,
    which is what a real pathway signal looks like.

    This function never sees the model. It operates purely on the DE
    table and the adjacency, so it's the model-agnostic baseline against
    which model-based pathway rankings should be compared. If the model's
    ranking doesn't beat this one, the model isn't adding pathway-level
    information.

    Parameters
    ----------
    de : pandas.DataFrame
        A per-gene DE table with at least the ``statistic`` column.
        Typically the output of ``differential_expression``. Its index
        should be gene names when ``adj`` is a DataFrame indexed by gene
        names — the two are aligned by name in that case. If both are
        integer-indexed, positional alignment is used and the caller is
        responsible for matching gene order.
    adj : torch.Tensor, numpy.ndarray, or pandas.DataFrame
        Binary gene-to-pathway adjacency of shape (n_genes, n_pathways).
        When a DataFrame, its index is gene names and columns are pathway
        names — both are used for alignment and in the returned DataFrame.
    statistic : str, default "lfc_mean"
        Column of ``de`` to use as the per-gene effect. Common choices:
        "lfc_mean" (the mean log2 fold change), "bf" (the log Bayes
        factor), or any other numeric column in ``de``.
    min_genes : int, default 5
        Pathways with fewer than this many member genes (after removing
        NaN statistic values) are skipped. Their row appears in the
        output with NaN pvalue and qvalue. Small pathways are too noisy
        for Mann-Whitney to say anything reliable.
    center : bool, default True
        When True, subtract the median of the statistic across all genes
        before running any test. This matters when the DE statistic comes
        from a compositional decoder (softmax over genes): one strongly
        induced gene mechanically depresses every other gene, and a
        genuinely up-regulated pathway can then have a negative raw
        median without the centering. Turn off only if you already know
        the panel-wide median is meaningful zero.

    Returns
    -------
    pandas.DataFrame
        One row per pathway, indexed by pathway name. Sorted by ``qvalue``
        ascending, then ``|effect|`` descending. Columns:

        - ``n_genes`` : number of annotated member genes for this pathway.
        - ``n_tested`` : members with a finite statistic value (subject to
          ``min_genes``).
        - ``median_member`` : median of the statistic across members.
        - ``median_reference`` : median of the statistic across non-members.
        - ``effect`` : ``median_member - median_reference``. Sign
          convention: positive means members are higher than the panel;
          negative means members are lower.
        - ``pvalue`` : two-sided Mann-Whitney p-value for the null that
          member and non-member statistics have the same distribution.
        - ``qvalue`` : Benjamini-Hochberg FDR-adjusted p-value over all
          tested pathways. Skipped pathways contribute NaN and do not
          inflate the adjustment burden.
        - ``sign`` : +1 if effect > 0, -1 if effect < 0, 0 otherwise.

    Raises
    ------
    ValueError
        If ``statistic`` is not a column of ``de``, or ``adj`` shape
        doesn't match ``de`` length.

    Notes
    -----
    Named indices are the safest way to pass ``de`` and ``adj`` here.
    ``differential_expression`` sorts its output by ``proba_de`` desc,
    so passing that sorted DataFrame with a gene-name index and an
    ``adj`` DataFrame with gene-name index does the right thing (they
    align by name, order doesn't matter). Passing bare arrays or
    integer-indexed DataFrames does positional alignment — the caller
    then owns keeping gene order consistent.

    Median-based effect and Mann-Whitney U were chosen together for
    robustness to outliers on both sides. If the mean were used for
    ``effect``, one huge member gene would drag the effect toward its
    sign regardless of how the other members behaved — the same failure
    mode this function exists to sidestep.
    """
    import numpy as np
    import pandas as pd
    import torch
    from scipy import stats

    # --- Coerce adj to (values, pathway_names, gene_names) ---
    if isinstance(adj, pd.DataFrame):
        pathway_names = list(adj.columns)
        adj_gene_names = list(adj.index)
        adj_arr = adj.values
    elif isinstance(adj, np.ndarray):
        adj_arr = adj
        pathway_names = [f"pathway_{j}" for j in range(adj.shape[1])]
        adj_gene_names = None
    elif isinstance(adj, torch.Tensor):
        adj_arr = adj.detach().cpu().numpy()
        pathway_names = [f"pathway_{j}" for j in range(adj.shape[1])]
        adj_gene_names = None
    else:
        raise TypeError(
            f"adj must be pandas.DataFrame, numpy.ndarray or torch.Tensor; "
            f"got {type(adj).__name__}"
        )
    adj_arr = np.asarray(adj_arr, dtype=np.float32)

    # --- Validate statistic column ---
    if statistic not in de.columns:
        raise ValueError(
            f"statistic={statistic!r} is not a column of `de`. "
            f"Available: {list(de.columns)}"
        )

    # --- Align de to adj's gene axis ---
    if adj_gene_names is not None and not isinstance(de.index, pd.RangeIndex):
        # Both named: align by name. This is the safe path.
        de_aligned = de.reindex(adj_gene_names)
        # Refuse silently-invalid alignment: if adj references genes de doesn't
        # have, reindex fills them with NaN — potentially many, changing the test.
        n_new_na = de_aligned[statistic].isna().sum() - de[statistic].isna().sum()
        if n_new_na > 0:
            missing = set(adj_gene_names) - set(de.index)
            raise ValueError(
                f"adj references {len(missing)} gene(s) not in `de`: "
                f"e.g. {sorted(missing)[:5]}"
            )
    else:
        # Positional alignment. Caller owns gene order.
        if len(de) != adj_arr.shape[0]:
            raise ValueError(
                f"de has {len(de)} genes but adj has {adj_arr.shape[0]}. "
                f"When indices don't match by name, positional alignment "
                f"requires equal lengths."
            )
        de_aligned = de

    stat_values = de_aligned[statistic].to_numpy(dtype=np.float64, copy=True)
    n_genes, n_pathways = adj_arr.shape

    # --- Optional median centering (compositional-decoder fix) ---
    finite = np.isfinite(stat_values)
    if center and finite.any():
        panel_median = np.median(stat_values[finite])
        stat_values = stat_values - panel_median
        finite = np.isfinite(stat_values)  # panel_median finite, so still valid

    # --- Per-pathway competitive Mann-Whitney ---
    n_genes_per_pathway = adj_arr.sum(axis=0).astype(int)
    median_member = np.full(n_pathways, np.nan, dtype=np.float64)
    median_reference = np.full(n_pathways, np.nan, dtype=np.float64)
    effect = np.full(n_pathways, np.nan, dtype=np.float64)
    pvalue = np.full(n_pathways, np.nan, dtype=np.float64)
    n_tested = np.zeros(n_pathways, dtype=np.int64)

    for j in range(n_pathways):
        member_mask = adj_arr[:, j].astype(bool)
        m_vals = stat_values[member_mask & finite]
        r_vals = stat_values[(~member_mask) & finite]
        n_tested[j] = len(m_vals)

        if len(m_vals) < min_genes:
            continue
        if len(r_vals) == 0:
            # Degenerate: pathway includes every gene in the panel.
            continue

        median_member[j] = float(np.median(m_vals))
        median_reference[j] = float(np.median(r_vals))
        effect[j] = median_member[j] - median_reference[j]

        # Two-sided Mann-Whitney U.
        try:
            _u, p = stats.mannwhitneyu(m_vals, r_vals, alternative="two-sided")
            pvalue[j] = float(p)
        except ValueError:
            # scipy raises when both sides are identical constants. Leave
            # pvalue NaN — the pathway is uninformative.
            pass

    # --- BH-adjusted q-values ---
    qvalue = _benjamini_hochberg(pvalue)

    sign = np.where(
        np.isnan(effect),
        0,
        np.where(effect > 0, 1, np.where(effect < 0, -1, 0)),
    ).astype(int)

    df = pd.DataFrame(
        {
            "n_genes": n_genes_per_pathway,
            "n_tested": n_tested,
            "median_member": median_member,
            "median_reference": median_reference,
            "effect": effect,
            "pvalue": pvalue,
            "qvalue": qvalue,
            "sign": sign,
        },
        index=pd.Index(pathway_names, name="pathway"),
    )

    # Sort by qvalue asc, then |effect| desc — same tie-breaking pattern
    # as differential_expression. NaN qvalues sort to the bottom.
    df = df.assign(_abs_effect=lambda d: d["effect"].abs())
    df = df.sort_values(
        by=["qvalue", "_abs_effect"],
        ascending=[True, False],
        kind="mergesort",
        na_position="last",
    ).drop(columns="_abs_effect")

    return df

def pseudobulk_paired_test(
    counts,
    donor_ids,
    condition,
    label_a,
    label_b,
    *,
    gene_names=None,
    min_donors: int = 3,
    pseudocount: float = 1.0,
    target_sum: float = 1e6,
) -> "pd.DataFrame":
    """Paired Wilcoxon per gene between two conditions, aggregated per donor.

    Fixes the pseudo-replication problem in cell-level differential-expression
    tests (Squair et al. 2021): a Wilcoxon over cells treats every cell as an
    independent observation, so 24,673 cells from 8 donors report the same
    statistical power as 24,673 truly-independent samples. That's not what's
    happening: cells within a donor share genetic background, batch and
    library-prep, so they are not independent. Every p-value comes out
    understated, hundreds of genes print as adjusted-p = 0 regardless of
    effect size, and the ranking becomes uninformative.

    The fix is to collapse each donor to a single pseudobulk profile per
    condition, then run a paired test across donors. The unit of independence
    then matches the unit of experimental replication.

    Pipeline
    --------
    1. Sum each donor's cell-level counts, per condition, to get one
       pseudobulk profile per (donor, condition) pair (Squair et al.'s
       recommended pooling — sum, not mean).
    2. Library-size normalise each pseudobulk sample to ``target_sum`` per
       cell (default: 1e6 → counts-per-million) so donors with different
       total mRNA output are comparable.
    3. Compute per-donor, per-gene log2 fold change:
           lfc[donor, gene] = log2((cpm_a + pseudocount) /
                                   (cpm_b + pseudocount))
       The pseudocount avoids log(0) when a gene is unobserved in one
       condition for a donor.
    4. Keep only donors present in both conditions (this is what makes
       the Wilcoxon *paired*).
    5. Per gene, run a two-sided paired Wilcoxon on the vector of
       per-donor log-fold-changes.
    6. BH-adjust the p-values across genes.
    7. Sort by ``pvalue`` asc, then ``|lfc_mean|`` desc. This matters at
       small n: at n=8 donors, the exact-null two-sided Wilcoxon
       distribution has a discrete floor of ``2 / 2^n = 2/256 ~ 0.0078``,
       so hundreds of genes will hit exactly this p-value at once. Sorting
       by p alone would leave them in gene-matrix column order, which is
       not an order. ``|lfc_mean|`` breaks the tie by effect size.

    Parameters
    ----------
    counts : array-like of shape (n_cells, n_genes)
        Raw count matrix. Any dtype accepted by numpy.asarray (torch tensor,
        numpy array, pandas DataFrame, scipy sparse). Cell-level counts;
        the pooling happens here.
    donor_ids : array-like of length n_cells
        Donor identifier per cell (any hashable type — strings usually).
    condition : array-like of length n_cells
        Condition label per cell.
    label_a, label_b : hashable
        The two condition labels to compare. Values in ``condition`` must
        contain both.
    gene_names : list of str, optional
        Gene labels for the returned DataFrame's index. If None, integer
        indices 0..n_genes-1 are used.
    min_donors : int, default 3
        Minimum number of donors that must have data in BOTH conditions
        for the test to run at all. Below this, raises rather than
        producing meaningless numbers. The paired Wilcoxon on n < 3 donors
        cannot reach conventional significance thresholds even with a
        perfectly separating effect, so refusing early is more useful than
        emitting a table of NaN.
    pseudocount : float, default 1.0
        Added to cpm values before the log-ratio, so a gene with zero
        counts in one donor-condition pair produces a finite (though
        capped) log fold change rather than -inf or +inf.
    target_sum : float, default 1e6
        Per-donor per-condition library-size normalisation target
        (counts-per-million by default).

    Returns
    -------
    pandas.DataFrame
        One row per gene, indexed by gene name (or integer 0..n_genes-1).
        Sorted by ``pvalue`` ascending, then ``|lfc_mean|`` descending.
        Columns:

        - ``n_donors_paired`` : donors present in both conditions.
        - ``lfc_mean`` : mean of per-donor log2 fold changes.
        - ``lfc_median`` : median of per-donor log2 fold changes (more
          robust to a single outlier donor).
        - ``pvalue`` : two-sided paired Wilcoxon p-value.
        - ``qvalue`` : BH-adjusted p-value.
        - ``sign`` : +1 if lfc_mean > 0, -1 if < 0, 0 otherwise.

    Raises
    ------
    ValueError
        If ``counts.shape[0]`` doesn't match len(donor_ids) or
        len(condition); if either label is absent from ``condition``; if
        fewer than ``min_donors`` donors appear in both conditions.

    Notes
    -----
    Why not just use scanpy's rank_genes_groups per donor? Because that
    still ends up cell-level. The pooling has to happen before the test,
    not after — that's the whole point of pseudobulk.

    Why library-size normalise before taking the ratio, rather than
    letting the ratio absorb it? Because a donor whose condition-A
    pseudobulk is 3x larger than their condition-B pseudobulk shouldn't
    push every gene's ratio up by 3x — that's a technical artifact of
    total sequencing depth, not biology.

    Why sum (not mean) for the pseudobulk aggregation? Because the sum
    is what a re-run of bulk RNA-seq on that donor's condition-A cells
    would measure. Mean would collapse the per-cell variance into
    a per-donor mean and understate uncertainty.

    Why paired Wilcoxon rather than paired t-test? No distributional
    assumption on the log-fold-changes; robust to a single outlier
    donor; matches Squair et al. 2021's recommendation.
    """
    import numpy as np
    import pandas as pd
    from scipy import stats

    # --- Coerce counts to a plain numpy array ---
    try:
        import torch
        if isinstance(counts, torch.Tensor):
            counts_arr = counts.detach().cpu().numpy()
        else:
            counts_arr = None
    except ImportError:
        counts_arr = None
    if counts_arr is None:
        if isinstance(counts, pd.DataFrame):
            counts_arr = counts.values
        else:
            try:
                from scipy import sparse
                if sparse.issparse(counts):
                    counts_arr = counts.toarray()
                else:
                    counts_arr = np.asarray(counts)
            except ImportError:
                counts_arr = np.asarray(counts)
    counts_arr = np.asarray(counts_arr, dtype=np.float64)

    n_cells, n_genes = counts_arr.shape

    # --- Shape checks on the alignment vectors ---
    donor_ids = np.asarray(donor_ids)
    condition = np.asarray(condition)
    if donor_ids.shape[0] != n_cells:
        raise ValueError(
            f"donor_ids has length {donor_ids.shape[0]}; "
            f"expected {n_cells} to match counts.shape[0]."
        )
    if condition.shape[0] != n_cells:
        raise ValueError(
            f"condition has length {condition.shape[0]}; "
            f"expected {n_cells} to match counts.shape[0]."
        )

    # --- Label existence in the data ---
    unique_labels = set(condition.tolist())
    if label_a not in unique_labels:
        raise ValueError(
            f"label_a={label_a!r} not present in `condition`. "
            f"Present labels: {sorted(unique_labels)}"
        )
    if label_b not in unique_labels:
        raise ValueError(
            f"label_b={label_b!r} not present in `condition`. "
            f"Present labels: {sorted(unique_labels)}"
        )

    # --- Pseudobulk: sum cell counts per (donor, condition) ---
    # Build a lookup {(donor, condition): pseudobulk vector} using boolean masks.
    unique_donors = np.unique(donor_ids)
    pseudobulk_a = {}
    pseudobulk_b = {}
    for donor in unique_donors:
        mask_a = (donor_ids == donor) & (condition == label_a)
        mask_b = (donor_ids == donor) & (condition == label_b)
        if mask_a.any():
            pseudobulk_a[donor] = counts_arr[mask_a].sum(axis=0)
        if mask_b.any():
            pseudobulk_b[donor] = counts_arr[mask_b].sum(axis=0)

    # --- Keep only donors present in BOTH conditions (this is the paired part) ---
    paired_donors = sorted(set(pseudobulk_a.keys()) & set(pseudobulk_b.keys()))
    n_paired = len(paired_donors)

    if n_paired < min_donors:
        raise ValueError(
            f"Only {n_paired} donor(s) present in both conditions "
            f"({label_a!r} and {label_b!r}); need at least min_donors={min_donors}. "
            f"A paired Wilcoxon on fewer donors cannot reach conventional "
            f"significance thresholds even with a perfectly separating effect."
        )

    # --- Stack the paired pseudobulks into (n_donors, n_genes) matrices ---
    mat_a = np.stack([pseudobulk_a[d] for d in paired_donors])  # (n_paired, n_genes)
    mat_b = np.stack([pseudobulk_b[d] for d in paired_donors])

    # --- Library-size normalise each row to target_sum ---
    lib_a = mat_a.sum(axis=1, keepdims=True)
    lib_b = mat_b.sum(axis=1, keepdims=True)
    # Guard against a donor with zero total counts in one condition (shouldn't
    # happen for real Kang data, but a fixture could hit this).
    lib_a = np.where(lib_a > 0, lib_a, 1.0)
    lib_b = np.where(lib_b > 0, lib_b, 1.0)
    cpm_a = mat_a / lib_a * target_sum
    cpm_b = mat_b / lib_b * target_sum

    # --- Per-donor per-gene log2 fold change ---
    lfc = np.log2((cpm_a + pseudocount) / (cpm_b + pseudocount))  # (n_paired, n_genes)

    # --- Per-gene mean, median, and paired Wilcoxon ---
    lfc_mean = lfc.mean(axis=0)
    lfc_median = np.median(lfc, axis=0)
    pvalue = np.full(n_genes, np.nan, dtype=np.float64)

    for g in range(n_genes):
        diffs = lfc[:, g]
        try:
            # scipy.stats.wilcoxon on the differences directly. Two-sided default.
            # For all-zero diffs, wilcoxon raises with newer scipy or returns NaN;
            # we catch and leave the entry NaN.
            _stat, p = stats.wilcoxon(diffs, alternative="two-sided")
            pvalue[g] = float(p)
        except (ValueError, RuntimeWarning):
            pass

    # --- BH-adjusted q-values (uses _benjamini_hochberg from Phase 1f-iii) ---
    qvalue = _benjamini_hochberg(pvalue)

    sign = np.where(
        np.isnan(lfc_mean),
        0,
        np.where(lfc_mean > 0, 1, np.where(lfc_mean < 0, -1, 0)),
    ).astype(int)

    # --- Build the DataFrame ---
    if gene_names is None:
        index = pd.RangeIndex(n_genes, name="gene")
    else:
        if len(gene_names) != n_genes:
            raise ValueError(
                f"gene_names has length {len(gene_names)}, expected n_genes={n_genes}"
            )
        index = pd.Index(gene_names, name="gene")

    df = pd.DataFrame(
        {
            "n_donors_paired": np.full(n_genes, n_paired, dtype=np.int64),
            "lfc_mean": lfc_mean,
            "lfc_median": lfc_median,
            "pvalue": pvalue,
            "qvalue": qvalue,
            "sign": sign,
        },
        index=index,
    )

    # Sort by pvalue asc, then |lfc_mean| desc — because of the 2/256 floor at
    # n=8 donors, hundreds of genes will tie on pvalue and need |lfc_mean| to
    # separate them.
    df = df.assign(_abs_lfc=lambda d: d["lfc_mean"].abs())
    df = df.sort_values(
        by=["pvalue", "_abs_lfc"],
        ascending=[True, False],
        kind="mergesort",
        na_position="last",
    ).drop(columns="_abs_lfc")

    return df


