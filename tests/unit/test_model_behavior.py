import numpy as np
import torch
import pyvae as mod


def test_reparameterise_has_no_hidden_scale():
    # Regression for the ELBO-consistency fix (#1): the sampling std must be
    # exactly exp(0.5 * log_var) -- no secret factor (e.g. * 0.01) -- so that
    # the KL term and the sampled posterior describe the same distribution.
    torch.manual_seed(0)
    adj = torch.tensor([[1, 0], [0, 1], [1, 0]], dtype=torch.float32)
    model = mod.InformedVAE(adj=adj, latent_dim=2, seed=0)

    n = 200_000
    mu = torch.zeros(n, 2)

    # log_var = 0  ->  std == 1
    z = model.reparameterise(mu, torch.zeros(n, 2))
    assert torch.allclose(z.mean(0), torch.zeros(2), atol=2e-2)
    assert torch.allclose(z.std(0), torch.ones(2), atol=2e-2)

    # log_var = 2*log(0.5)  ->  std == 0.5  (confirms the exp(0.5*log_var) form)
    log_var = torch.full((n, 2), 2.0 * float(np.log(0.5)))
    z = model.reparameterise(mu, log_var)
    assert torch.allclose(z.std(0), torch.full((2,), 0.5), atol=2e-2)
