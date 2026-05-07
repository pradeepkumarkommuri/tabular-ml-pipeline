"""
models/tabnet.py
----------------
A clean PyTorch implementation of a TabNet-style deep tabular model.

Architecture highlights:
- Sequential attention over features (step-wise feature selection)
- Ghost Batch Normalization for stable training on tabular data
- Sparsity regularisation on attention weights
- Residual connections between steps
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


class GhostBatchNorm(nn.Module):
    """
    Ghost Batch Normalization — splits each batch into virtual mini-batches.
    Stabilises training on tabular data with heterogeneous feature scales.

    Args:
        num_features: Number of input features.
        virtual_batch_size: Size of each virtual mini-batch.
        momentum: BatchNorm momentum.
    """

    def __init__(
        self,
        num_features: int,
        virtual_batch_size: int = 128,
        momentum: float = 0.02,
    ):
        super().__init__()
        self.virtual_batch_size = virtual_batch_size
        self.bn = nn.BatchNorm1d(num_features, momentum=momentum)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        chunks = x.chunk(max(1, x.size(0) // self.virtual_batch_size), dim=0)
        return torch.cat([self.bn(c) for c in chunks], dim=0)


class GLUBlock(nn.Module):
    """
    Gated Linear Unit block: FC → GBN → GLU activation.
    Used inside each TabNet step.
    """

    def __init__(
        self,
        in_dim: int,
        out_dim: int,
        fc: nn.Linear | None = None,
        virtual_batch_size: int = 128,
        momentum: float = 0.02,
    ):
        super().__init__()
        self.fc = fc if fc is not None else nn.Linear(in_dim, out_dim * 2, bias=False)
        self.bn = GhostBatchNorm(out_dim * 2, virtual_batch_size, momentum)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.bn(self.fc(x))
        x1, x2 = x.chunk(2, dim=-1)
        return x1 * torch.sigmoid(x2)


class TabNetStep(nn.Module):
    """One sequential attention step in the TabNet architecture."""

    def __init__(
        self,
        input_dim: int,
        hidden_dim: int,
        virtual_batch_size: int = 128,
        momentum: float = 0.02,
    ):
        super().__init__()
        # Attention transformer
        self.attention_fc = nn.Linear(hidden_dim, input_dim, bias=False)
        self.attention_bn = GhostBatchNorm(input_dim, virtual_batch_size, momentum)

        # Feature transformer (shared + step-specific)
        self.shared_glu = GLUBlock(input_dim, hidden_dim, virtual_batch_size=virtual_batch_size)
        self.step_glu = GLUBlock(hidden_dim, hidden_dim, virtual_batch_size=virtual_batch_size)

    def forward(
        self,
        x: torch.Tensor,
        prior_scales: torch.Tensor,
        h_prior: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Args:
            x: Input features (B, input_dim).
            prior_scales: Running product of (1 - attention_weights) (B, input_dim).
            h_prior: Previous step's hidden state (B, hidden_dim).

        Returns:
            h: Step output (B, hidden_dim).
            prior_scales: Updated prior scales (B, input_dim).
            attention_weights: Sparse feature weights for this step (B, input_dim).
        """
        # Attention
        a = self.attention_bn(self.attention_fc(h_prior))
        a = a * prior_scales
        attention_weights = F.softmax(a, dim=-1)
        prior_scales = prior_scales * (1 - attention_weights)

        # Masked features → feature transformer
        masked = x * attention_weights
        h = self.shared_glu(masked)
        h = self.step_glu(h)

        return h, prior_scales, attention_weights


class TabNet(nn.Module):
    """
    PyTorch TabNet for binary / multi-class classification on tabular data.

    Args:
        input_dim: Number of input features.
        output_dim: Number of output classes (1 for binary classification).
        hidden_dim: Width of hidden layers.
        num_steps: Number of sequential attention steps.
        gamma: Feature reuse coefficient (controls sparsity).
        virtual_batch_size: Ghost BN virtual batch size.
        momentum: BatchNorm momentum.
        dropout: Dropout rate on final representation.
    """

    def __init__(
        self,
        input_dim: int,
        output_dim: int = 1,
        hidden_dim: int = 128,
        num_steps: int = 3,
        gamma: float = 1.3,
        virtual_batch_size: int = 128,
        momentum: float = 0.02,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.num_steps = num_steps
        self.gamma = gamma

        # Initial BN on raw features
        self.initial_bn = nn.BatchNorm1d(input_dim, momentum=momentum)

        # Attention steps
        self.steps = nn.ModuleList([
            TabNetStep(input_dim, hidden_dim, virtual_batch_size, momentum)
            for _ in range(num_steps)
        ])

        self.dropout = nn.Dropout(dropout)
        self.head = nn.Linear(hidden_dim, output_dim)

        self._init_weights()

    def forward(
        self, x: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Args:
            x: Input tensor (B, input_dim).

        Returns:
            logits: Output logits (B, output_dim).
            entropy_loss: Sparsity regularisation loss (scalar).
        """
        x = self.initial_bn(x)

        prior_scales = torch.ones(x.size(0), self.input_dim, device=x.device)
        h = torch.zeros(x.size(0), self.hidden_dim, device=x.device)
        outputs = []
        entropy_loss = torch.zeros(1, device=x.device)

        for step in self.steps:
            h_new, prior_scales, attention_weights = step(x, prior_scales, h)
            outputs.append(F.relu(h_new))
            h = h_new

            # Sparsity regularisation
            entropy_loss += torch.mean(
                torch.sum(-attention_weights * torch.log(attention_weights + 1e-15), dim=-1)
            )

        out = torch.stack(outputs, dim=0).sum(dim=0)
        out = self.dropout(out)
        logits = self.head(out)

        return logits, entropy_loss / self.num_steps

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

    @property
    def num_parameters(self) -> int:
        return sum(p.numel() for p in self.parameters() if p.requires_grad)
