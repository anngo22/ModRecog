import numpy as np
import torch
from omegaconf import OmegaConf

from modrecog.model import ModRecogLightningModule, ResNet1D


def _make_cfg():
    return OmegaConf.create(
        {
            "model": {
                "num_classes": 24,
                "input_channels": 2,
                "num_residual_stacks": 6,
                "base_filters": 32,
                "kernel_size": 3,
                "alpha_dropout_rate": 0.1,
            },
            "data": {"window_length": 1024},
            "training": {"lr": 1e-3},
            "logging": {
                "experiment_name": "test",
                "tracking_uri": "http://localhost:8080",
                "tags": {},
                "log_model": False,
            },
            "seed": 42,
        }
    )


def test_resnet1d_output_shape():
    model = ResNet1D(
        num_classes=24,
        input_channels=2,
        num_residual_stacks=6,
        base_filters=32,
        kernel_size=3,
        alpha_dropout_rate=0.1,
        window_length=1024,
    )
    x = torch.randn(4, 2, 1024)
    out = model(x)
    assert out.shape == (4, 24)


def test_lightning_module_forward():
    cfg = _make_cfg()
    module = ModRecogLightningModule(cfg)
    x = torch.randn(2, 2, 1024)
    out = module(x)
    assert out.shape == (2, 24)


def test_dataset_transpose():
    raw = np.random.randn(1024, 2).astype(np.float32)
    x = torch.from_numpy(raw).T.contiguous()
    assert x.shape == (2, 1024), f"Expected (2,1024), got {x.shape}"


def test_parameter_count():
    model = ResNet1D(
        num_classes=24,
        input_channels=2,
        num_residual_stacks=6,
        base_filters=32,
        kernel_size=3,
        alpha_dropout_rate=0.1,
        window_length=1024,
    )
    n_params = sum(p.numel() for p in model.parameters())

    assert 100_000 < n_params < 400_000, f"Unexpected param count: {n_params}"
