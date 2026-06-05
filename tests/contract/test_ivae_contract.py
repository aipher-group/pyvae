import inspect

import numpy as np
import pandas as pd
import torch
import pyvae as mod


def test_public_api_exists():
    assert hasattr(mod, "InformedLinear")
    assert hasattr(mod, "InformedVAE")
    assert hasattr(mod, "train_ivae")


def test_train_signature_contract():
    sig = inspect.signature(mod.train_ivae)
    expected = [
        "model",
        "x_train",
        "x_val",
        "epochs",
        "batch_size",
        "patience",
        "lr",
        "device",
    ]
    assert list(sig.parameters.keys()) == expected


def test_informed_linear_mask_behavior():
    adj = torch.tensor([[1, 0], [0, 1], [1, 0]], dtype=torch.float32)
    layer = mod.InformedLinear(adj, activation="linear")

    with torch.no_grad():
        layer.weight.fill_(1.0)
        layer.bias.zero_()

    x = torch.tensor([[1.0, 2.0, 3.0]])
    out = layer(x)
    assert out.shape == (1, 2)
    assert torch.allclose(out, torch.tensor([[4.0, 2.0]]), atol=1e-6)


def test_model_forward_and_loss_shapes():
    torch.manual_seed(0)
    adj = torch.tensor(
        [[1, 0, 1], [0, 1, 1], [1, 1, 0], [0, 1, 0]], dtype=torch.float32
    )
    model = mod.InformedVAE(adj=adj, latent_dim=2, seed=0)

    x = torch.randn(5, 4)
    recon, mu, log_var, h = model(x)
    assert recon.shape == x.shape
    assert mu.shape == (5, 2)
    assert log_var.shape == (5, 2)
    assert h.shape == (5, 3)

    loss = model.loss(x, recon, mu, log_var, h)
    assert loss.ndim == 0
    assert torch.isfinite(loss)
    assert float(loss.detach()) > 0


def test_train_returns_history_lists():
    torch.manual_seed(0)
    np.random.seed(0)

    n_samples = 24
    n_genes = 6
    x_train = pd.DataFrame(np.random.randn(n_samples, n_genes).astype(np.float32))
    x_val = pd.DataFrame(np.random.randn(n_samples, n_genes).astype(np.float32))
    adj = torch.tensor(np.random.randint(0, 2, size=(n_genes, 4)), dtype=torch.float32)
    model = mod.InformedVAE(adj=adj, latent_dim=2, seed=0)

    trained, history = mod.train_ivae(
        model,
        x_train,
        x_val,
        epochs=3,
        batch_size=8,
        patience=3,
        lr=1e-3,
        device="cpu",
    )

    assert isinstance(history, dict)
    assert "train" in history and "val" in history
    assert len(history["train"]) >= 1
    assert len(history["val"]) >= 1
    assert len(history["train"]) == len(history["val"])
    assert isinstance(trained, mod.InformedVAE)