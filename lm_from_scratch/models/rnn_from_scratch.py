"""A hand-rolled GRU cell.

PyTorch's ``nn.GRUCell`` is correct and fast. This module reimplements
the gate equations explicitly so the reader can compare them to the
reference implementation. The tests in ``tests/test_rnn_lm.py`` verify
that with copied weights the two cells produce numerically equivalent
outputs on a fixed input.
"""

from __future__ import annotations

import torch
import torch.nn as nn


class HandRolledGRUCell(nn.Module):
    r"""GRU cell, equations spelled out.

    Update gate :math:`z`, reset gate :math:`r` and candidate hidden state
    :math:`\tilde h`:

    .. math::

        z_t = \sigma(W_z x_t + U_z h_{t-1}) \\
        r_t = \sigma(W_r x_t + U_r h_{t-1}) \\
        \tilde h_t = \tanh(W_h x_t + r_t \odot (U_h h_{t-1})) \\
        h_t = (1 - z_t) \odot \tilde h_t + z_t \odot h_{t-1}

    The weight layout (``weight_ih`` and ``weight_hh``, stacked
    ``r``-then-``z``-then-``n``) and the role of :math:`z_t` follow
    PyTorch's ``nn.GRUCell``: when :math:`z_t = 1` the hidden state is
    kept unchanged, when :math:`z_t = 0` it is replaced by the candidate.
    (The original Cho et al. 2014 paper uses the opposite convention.)
    """

    def __init__(self, input_size: int, hidden_size: int) -> None:
        super().__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        # Stacked weight layout matching PyTorch's GRUCell: rows are r, z, n.
        self.weight_ih = nn.Parameter(torch.empty(3 * hidden_size, input_size))
        self.weight_hh = nn.Parameter(torch.empty(3 * hidden_size, hidden_size))
        self.bias_ih = nn.Parameter(torch.zeros(3 * hidden_size))
        self.bias_hh = nn.Parameter(torch.zeros(3 * hidden_size))
        self.reset_parameters()

    def reset_parameters(self) -> None:
        stdv = 1.0 / (self.hidden_size**0.5)
        for w in (self.weight_ih, self.weight_hh):
            nn.init.uniform_(w, -stdv, stdv)

    def forward(self, x: torch.Tensor, h: torch.Tensor) -> torch.Tensor:
        H = self.hidden_size
        # Linear projections of input and hidden, split into three gates.
        gi = x @ self.weight_ih.t() + self.bias_ih
        gh = h @ self.weight_hh.t() + self.bias_hh
        i_r, i_z, i_n = gi.chunk(3, dim=-1)
        h_r, h_z, h_n = gh.chunk(3, dim=-1)
        r = torch.sigmoid(i_r + h_r)
        z = torch.sigmoid(i_z + h_z)
        n = torch.tanh(i_n + r * h_n)
        return (1 - z) * n + z * h
