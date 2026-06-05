# pyvae - Student Edition

Implement an **Informed Variational Autoencoder (iVAE)** in PyTorch.
The encoder is biologically constrained: each gene can only influence the
pathways that contain it, according to the [Reactome](https://reactome.org/)
database.

---

## 1  Background

### What is a VAE?

A Variational Autoencoder learns a *compressed* (latent) representation $z$
of the input $x$ by maximising the Evidence Lower Bound (ELBO):

$$
\text{ELBO} = \mathbb{E}_{q(z|x)}[\log p(x|z)] - \text{KL}(q(z|x) \,\|\, p(z))
$$

- **Reconstruction term** - the decoder should reproduce the input from $z$.
- **KL divergence** - the posterior $q(z|x)$ should stay close to the
  standard Gaussian prior $p(z) = \mathcal{N}(0, I)$.

The closed-form KL for Gaussian distributions is:

$$
\mathcal{L}_{\text{KL}} = -\frac{1}{2}\,\mathbb{E}\!\left[
  1 + \log\sigma^2 - \mu^2 - \sigma^2
\right]
$$

### Why biological constraints?

Gene expression is high-dimensional (~20 000 genes) but structured: genes
work together in *pathways*.  By restricting the encoder's first layer to
only allow gene -> pathway connections that exist in Reactome, we:

- reduce the number of parameters,
- make the latent representation interpretable (each node = a pathway),
- incorporate prior biological knowledge.

The constraint is enforced via a **binary mask** on the weight matrix:

$$
W_{\text{eff}} = W \odot M
$$

where $M_{ji} = 1$ only if gene $i$ belongs to pathway $j$, and $\odot$
denotes the element-wise (Hadamard) product.

### Reparameterisation trick

Sampling $z \sim \mathcal{N}(\mu, \sigma^2)$ is not differentiable.
We rewrite it as:

$$
z = \mu + \sigma\,\varepsilon, \qquad \varepsilon \sim \mathcal{N}(0, I)
$$

Gradients flow through $\mu$ and $\sigma$ but not through the stochastic
$\varepsilon$.

---

## 2  Setup

This project uses [pixi](https://prefix.dev/) for environment management.

```bash
# Install all dependencies (creates a local .pixi/ env)
pixi install

# Activate the environment
pixi shell
```

All further commands assume you are inside `pixi shell`.

---

## 3  Implementation steps

Work through the files in this order - each one builds on the previous.

### Step 1 - `pyvae/utils.py`

Implement `set_all_seeds(seed)`.  Set the seed for `random`, `numpy`,
`torch`, and (if available) `torch.cuda`.

### Step 2 - `pyvae/layers.py`

Implement `InformedLinear.__init__` and `InformedLinear.forward`.

Key points:
- Initialise `self.weight` with Xavier uniform (`nn.init.xavier_uniform_`).
- Register `adj.T` as a non-trainable buffer called `"mask"`.
- Register a `"bias_mask"` that is 1 for pathways with >= 1 gene input.
- In `forward`, apply $W_{\text{eff}} = W \odot M$ before the linear transform.

### Step 3 - `pyvae/models.py`

Implement the six methods of `InformedVAE`:

| Method | What to do |
|---|---|
| `__init__` | Create the five layers listed in the docstring |
| `encode` | genes -> pathways -> $(\mu,\, \log\sigma^2,\, h)$ |
| `reparameterise` | sample $z = \mu + \sigma\varepsilon$ via the reparameterisation trick |
| `decode` | $z$ -> pathways (tanh) -> genes (linear, no activation) |
| `forward` | chain encode -> reparameterise -> decode |
| `loss` | $\mathcal{L}_{\text{recon}} + \mathcal{L}_{\text{KL}} + \mathcal{L}_{L2}$ (see docstring) |

### Step 4 - `pyvae/train.py`

Implement `train_ivae`.  Follow the algorithm description in the docstring.
Pay attention to:
- gradient clipping after each backward pass,
- early stopping: reset the patience counter only when `val_loss` improves.

### Step 5 - `pyvae/datasets.py`  *(optional)*

Implement `load_kang` to download and preprocess the Kang PBMC dataset.
Only needed if you want to run the full pipeline on real data.

### Step 6 - `pyvae/bio.py`  *(optional)*

Implement the four helpers (`get_reactome_adj`, `get_reactome_hierarchical_adj`,
`sync_gexp_adj`, `build_model_config`) to load Reactome pathway data and build
the adjacency matrix.  Only needed for the full pipeline.

---

## 4  Testing your implementation

Use `pixi run pytest` instead of calling `pytest` directly.  `pixi run`
executes the command inside the managed environment, so it always picks up
the correct Python interpreter and installed packages — even if you have
other Python versions on your system or forgot to activate `pixi shell`.

```bash
# Run all contract tests (do this after every module you implement)
pixi run pytest tests/ -v

# Run a single test file
pixi run pytest tests/contract/test_ivae_contract.py -v

# Stop on first failure (useful while debugging one function at a time)
pixi run pytest tests/ -x
```

**When to run tests:**
- Before you write anything: all three implementation tests should raise
  `NotImplementedError` — this confirms the skeleton is intact.
- After each step: run the full suite to check you have not broken anything
  you already implemented.
- After Step 2 (`layers.py`): `test_informed_linear_mask_behavior` should pass.
- After Step 3 (`models.py`): `test_model_forward_and_loss_shapes` should pass.
- After Step 4 (`train.py`): `test_train_returns_history_lists` should pass.

The test suite (`tests/contract/test_ivae_contract.py`) checks:

| Test | What it verifies |
|---|---|
| `test_public_api_exists` | All public names are importable |
| `test_train_signature_contract` | `train_ivae` has the exact 8 expected parameters |
| `test_informed_linear_mask_behavior` | Masked weights produce correct outputs |
| `test_model_forward_and_loss_shapes` | Forward pass shapes; loss is a finite positive scalar |
| `test_train_returns_history_lists` | Training runs and returns matching train/val history |

The first two tests (`test_public_api_exists`, `test_train_signature_contract`)
pass immediately without any implementation — they only check that the
API structure is correct.  The remaining three require working code.

---

## 5  Full pipeline example

Once all steps are implemented:

```python
import pandas as pd
import torch
import pyvae

# 1. Load and preprocess data
adata = pyvae.load_kang(data_folder="./data", n_genes=3000)

# 2. Split into train / validation
train_mask = adata.obs["split"] == "train"
x_train = pd.DataFrame(adata[train_mask].X.toarray(), columns=adata.var_names)
x_val   = pd.DataFrame(adata[~train_mask].X.toarray(), columns=adata.var_names)

# 3. Build biological configuration
config = pyvae.build_model_config(
    genes=x_train.columns,
    model_kind="ivae_reactome_hierarchical_d2",
    resources_dir="./resources",
)

# 4. Create adjacency tensor
adj = torch.tensor(config.model_layer[0].values, dtype=torch.float32)

# 5. Initialise model
model = pyvae.InformedVAE(adj=adj, latent_dim=64, l2_lambda=1e-5)

# 6. Train
model, history = pyvae.train_ivae(
    model, x_train, x_val,
    epochs=500, batch_size=32, patience=50,
    lr=1e-4, device="cpu",
)

# 7. Encode new cells
x_tensor = torch.tensor(x_val.values, dtype=torch.float32)
z_mean, z_log_var, h = model.encode(x_tensor)
# z_mean : latent representations  (n_cells, latent_dim)
# h      : pathway activations     (n_cells, n_pathways) - biologically interpretable
```

---

## 6  Tips

- **Numerical stability**: always clamp `log_var` to $[-3, 3]$ before calling
  `exp`.  Unclamped log-variance can produce `NaN` in the very first epoch.
- **Gradient clipping**: the reconstruction loss is scaled by $n_{\text{genes}}$
  (~3 000), which can create large gradients early in training.
- **Loss scale**: it is normal for the total loss to start in the thousands
  and decrease over hundreds of epochs.  Do not worry if it looks large.
- **Variance scaling**: scaling $\sigma$ by a factor less than 1 keeps the
  initial KL term small, letting the model focus on reconstruction first.
- **Device mismatch**: make sure both the model and the input tensors are on
  the same device (`.to(device)`) before calling `forward` or `loss`.

---

## 7  Project structure

```
pyvae/
├── __init__.py      public API (do not modify)
├── utils.py         Step 1 - seed utilities
├── layers.py        Step 2 - InformedLinear
├── models.py        Step 3 - InformedVAE
├── train.py         Step 4 - training loop
├── datasets.py      Step 5 - Kang dataset loader
└── bio.py           Step 6 - Reactome adjacency helpers

tests/
└── contract/
    └── test_ivae_contract.py   <- run these to check your work
```
