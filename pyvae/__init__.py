from importlib.metadata import PackageNotFoundError
from importlib.metadata import version as _dist_version

from pyvae.bio import (
    InformedModelConfig,
    build_model_config,
    permute_adjacency,
    swap_condition,
    sync_gexp_adj,
)
from pyvae.datasets import load_kang
from pyvae.interpret import (
    bayes_factor_da,
    differential_expression,
    integrated_gradients,
    pathway_activity,
    pathway_unit_fidelity,
    pseudobulk_paired_test,
    unit_contrast,
    unit_sign_structure,
)
from pyvae.layers import InformedLinear
from pyvae.models import InformedVAE
from pyvae.train import train_ivae, train_ivae_modern
from pyvae.utils import set_all_seeds

__all__ = [
    "InformedLinear",
    "InformedVAE",
    "train_ivae",
    "train_ivae_modern",
    "load_kang",
    "build_model_config",
    "sync_gexp_adj",
    "InformedModelConfig",
    "set_all_seeds",
    "swap_condition",
    "bayes_factor_da",
    "differential_expression",
    "integrated_gradients",
    "pathway_activity",
    "pathway_unit_fidelity",
    "permute_adjacency",
    "pseudobulk_paired_test",
    "unit_contrast",
    "unit_sign_structure",
]


# Read from the installed distribution, so `pyproject.toml` stays the single
# place the number is written (`pixi run bump-version`). A source tree that was
# never installed has no distribution to read.
try:
    __version__ = _dist_version("pyvae")
except PackageNotFoundError:
    __version__ = "0+unknown"
