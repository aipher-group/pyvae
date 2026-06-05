"""
Biologically-constrained linear layer for the PyTorch iVAE.

InformedLinear enforces a binary adjacency mask on its weight matrix so that
gene i can only contribute to pathway j when adj[i, j] == 1 - mirroring
the biological knowledge encoded in the Reactome / D2-hierarchy database.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F  # noqa: F401 - students will use F.linear / F.relu


class InformedLinear(nn.Module):
    """
    Linear layer with a fixed binary sparsity mask.

    Standard nn.Linear allows every input neuron to connect to every
    output neuron.  InformedLinear restricts these connections using
    a biological prior: gene i may only influence pathway j if
    adj[i, j] == 1.

    Parameters
    ----------
    adj : torch.Tensor, shape (in_features, out_features)
        Binary indicator matrix.  adj[i, j] = 1 allows
        weight[j, i] to be non-zero (gene i -> pathway j connection).
        Rows are genes; columns are pathways.
    activation : str
        Non-linearity applied after the masked linear transformation.
        One of "tanh", "relu", or "linear" (no activation).

    Attributes to create
    --------------------
    weight : nn.Parameter, shape (out_features, in_features)
        Learnable weight matrix.  Initialise with nn.init.xavier_uniform_.
        Xavier init keeps activations in a healthy range at the start of
        training regardless of layer width.
    bias : nn.Parameter, shape (out_features,)
        Learnable bias, initialised to zero.
    mask : non-trainable buffer, shape (out_features, in_features)
        Transposed adjacency matrix (adj.T), cast to float.
        Stored as a buffer so it moves to GPU automatically with the model.
        We transpose because weight has layout (out, in).
    bias_mask : non-trainable buffer, shape (out_features,)
        1.0 where the pathway has at least one gene input, 0.0 otherwise.
        Prevents pathways with no inputs from learning a free bias offset.
    activation : str
        Stored activation name, used in forward.
    """

    def __init__(self, adj: torch.Tensor, activation: str = "tanh"):
        super().__init__()
        in_f, out_f = adj.shape  # provided for you
        self.activation = activation
        raise NotImplementedError

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Apply the masked linear transformation followed by the activation.

        The sparsity constraint is enforced via element-wise (Hadamard product)
          multiplication of the weight matrix with the mask:

            W_eff = W * M        (* means element-wise here)

        where M is the transposed adjacency matrix.

        The full forward pass is:

            out = activation( x @ W_eff.T + (b * b_mask) )

        where activation is tanh, relu, or identity.

        Parameters
        ----------
        x : torch.Tensor, shape (..., in_features)
            Input gene-expression batch.

        Returns
        -------
        torch.Tensor, shape (..., out_features)
            Pathway activations after the masked transformation.
        """
        # w: effective weight matrix after enforcing the sparsity mask
        # b: effective bias, zeroed out for pathways with no gene inputs
        # out: result of the linear transform, before activation
        raise NotImplementedError
