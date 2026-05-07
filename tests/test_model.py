"""
tests/test_model.py
-------------------
Unit tests for the TabNet model architecture.
"""

import pytest
import torch

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.models.tabnet import TabNet


class TestTabNet:
    @pytest.fixture
    def model(self):
        return TabNet(input_dim=10, output_dim=1, hidden_dim=32, num_steps=2)

    def test_forward_shape(self, model):
        x = torch.randn(16, 10)
        logits, entropy = model(x)
        assert logits.shape == (16, 1)
        assert entropy.ndim == 1

    def test_entropy_loss_is_positive(self, model):
        x = torch.randn(32, 10)
        _, entropy = model(x)
        assert entropy.item() >= 0

    def test_parameter_count_nonzero(self, model):
        assert model.num_parameters > 0

    def test_different_batch_sizes(self, model):
        # batch_size=1 requires eval mode (BatchNorm needs >1 sample during training)
        for batch_size in [4, 64, 256]:
            x = torch.randn(batch_size, 10)
            logits, _ = model(x)
            assert logits.shape[0] == batch_size

        model.eval()
        with torch.no_grad():
            x = torch.randn(1, 10)
            logits, _ = model(x)
            assert logits.shape[0] == 1

    def test_gradient_flow(self, model):
        x = torch.randn(8, 10)
        logits, entropy = model(x)
        loss = logits.mean() + entropy
        loss.backward()
        for name, param in model.named_parameters():
            if param.requires_grad:
                assert param.grad is not None, f"No gradient for {name}"

    def test_eval_mode_deterministic(self, model):
        model.eval()
        x = torch.randn(4, 10)
        with torch.no_grad():
            out1, _ = model(x)
            out2, _ = model(x)
        assert torch.allclose(out1, out2)

    def test_multi_class_output(self):
        model = TabNet(input_dim=8, output_dim=5, hidden_dim=16, num_steps=2)
        x = torch.randn(10, 8)
        logits, _ = model(x)
        assert logits.shape == (10, 5)
