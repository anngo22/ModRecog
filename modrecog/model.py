from typing import Any

import pytorch_lightning as pl
import torch
import torch.nn as nn
from omegaconf import DictConfig, OmegaConf
from torchmetrics import Accuracy, F1Score, Precision, Recall


class ResidualStack(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, kernel_size: int) -> None:
        super().__init__()
        pad = kernel_size // 2
        self.conv1 = nn.Conv1d(in_channels, out_channels, kernel_size, padding=pad)
        self.conv2 = nn.Conv1d(out_channels, out_channels, kernel_size, padding=pad)
        self.selu = nn.SELU()

        self.shortcut = (
            nn.Conv1d(in_channels, out_channels, kernel_size=1)
            if in_channels != out_channels
            else nn.Identity()
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        residual = self.shortcut(x)
        out = self.selu(self.conv1(x))
        out = self.conv2(out)
        return self.selu(out + residual)


class ResNet1D(nn.Module):
    def __init__(
        self,
        num_classes: int,
        input_channels: int,
        num_residual_stacks: int,
        base_filters: int,
        kernel_size: int,
        alpha_dropout_rate: float,
        window_length: int,
    ) -> None:
        super().__init__()

        stacks: list[nn.Module] = []
        in_ch = input_channels
        for i in range(num_residual_stacks):
            out_ch = base_filters * (2 ** min(i // 2, 2))
            stacks.append(ResidualStack(in_ch, out_ch, kernel_size))
            in_ch = out_ch

        self.residual_stacks = nn.Sequential(*stacks)
        self.global_avg_pool = nn.AdaptiveAvgPool1d(1)
        self.flatten = nn.Flatten()
        self.dropout = nn.AlphaDropout(alpha_dropout_rate)
        self.fc1 = nn.Linear(in_ch, 128)
        self.selu = nn.SELU()
        self.fc2 = nn.Linear(128, num_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.residual_stacks(x)
        x = self.global_avg_pool(x)
        x = self.flatten(x)
        x = self.dropout(x)
        x = self.selu(self.fc1(x))
        return self.fc2(x)


class ModRecogLightningModule(pl.LightningModule):
    def __init__(self, cfg: DictConfig | dict) -> None:
        super().__init__()

        if isinstance(cfg, dict):
            cfg = OmegaConf.create(cfg)

        self.save_hyperparameters(OmegaConf.to_container(cfg, resolve=True))
        self.cfg = cfg

        self.model = ResNet1D(
            num_classes=cfg.model.num_classes,
            input_channels=cfg.model.input_channels,
            num_residual_stacks=cfg.model.num_residual_stacks,
            base_filters=cfg.model.base_filters,
            kernel_size=cfg.model.kernel_size,
            alpha_dropout_rate=cfg.model.alpha_dropout_rate,
            window_length=cfg.data.window_length,
        )
        self.criterion = nn.CrossEntropyLoss()

        num_classes = cfg.model.num_classes
        self.train_acc = Accuracy(task="multiclass", num_classes=num_classes)
        self.val_acc = Accuracy(task="multiclass", num_classes=num_classes)
        self.test_acc = Accuracy(task="multiclass", num_classes=num_classes)
        self.val_f1 = F1Score(
            task="multiclass", num_classes=num_classes, average="macro"
        )
        self.test_f1 = F1Score(
            task="multiclass", num_classes=num_classes, average="macro"
        )
        self.val_top3 = Accuracy(task="multiclass", num_classes=num_classes, top_k=3)
        self.test_top3 = Accuracy(task="multiclass", num_classes=num_classes, top_k=3)
        self.val_precision = Precision(
            task="multiclass", num_classes=num_classes, average="macro"
        )
        self.test_precision = Precision(
            task="multiclass", num_classes=num_classes, average="macro"
        )
        self.val_recall = Recall(
            task="multiclass", num_classes=num_classes, average="macro"
        )
        self.test_recall = Recall(
            task="multiclass", num_classes=num_classes, average="macro"
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.model(x)

    def _shared_step(
        self, batch: tuple[torch.Tensor, torch.Tensor]
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        x, y = batch
        logits = self(x)
        loss = self.criterion(logits, y)
        preds = logits.argmax(dim=1)
        return loss, logits, preds, y

    def training_step(
        self, batch: tuple[torch.Tensor, torch.Tensor], batch_idx: int
    ) -> torch.Tensor:
        loss, _logits, preds, y = self._shared_step(batch)
        self.train_acc(preds, y)
        self.log("train/loss", loss, on_step=False, on_epoch=True, prog_bar=True)
        self.log(
            "train/acc", self.train_acc, on_step=False, on_epoch=True, prog_bar=True
        )
        return loss

    def validation_step(
        self, batch: tuple[torch.Tensor, torch.Tensor], batch_idx: int
    ) -> None:
        loss, logits, preds, y = self._shared_step(batch)
        self.val_acc(preds, y)
        self.val_f1(preds, y)
        self.val_top3(logits, y)
        self.val_precision(preds, y)
        self.val_recall(preds, y)
        self.log("val/loss", loss, on_epoch=True, prog_bar=True)
        self.log("val/acc", self.val_acc, on_epoch=True, prog_bar=True)
        self.log("val/f1", self.val_f1, on_epoch=True, prog_bar=True)
        self.log("val/top3_acc", self.val_top3, on_epoch=True)
        self.log("val/precision", self.val_precision, on_epoch=True)
        self.log("val/recall", self.val_recall, on_epoch=True)

    def test_step(
        self, batch: tuple[torch.Tensor, torch.Tensor], batch_idx: int
    ) -> None:
        loss, logits, preds, y = self._shared_step(batch)
        self.test_acc(preds, y)
        self.test_f1(preds, y)
        self.test_top3(logits, y)
        self.test_precision(preds, y)
        self.test_recall(preds, y)
        self.log("test/loss", loss)
        self.log("test/acc", self.test_acc)
        self.log("test/f1", self.test_f1)
        self.log("test/top3_acc", self.test_top3)
        self.log("test/precision", self.test_precision)
        self.log("test/recall", self.test_recall)

    def configure_optimizers(self) -> Any:
        optimizer = torch.optim.Adam(self.parameters(), lr=self.cfg.training.lr)
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, mode="min", patience=5, factor=0.5
        )
        return {
            "optimizer": optimizer,
            "lr_scheduler": {"scheduler": scheduler, "monitor": "val/loss"},
        }
