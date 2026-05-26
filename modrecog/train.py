import subprocess
from pathlib import Path

import hydra
import matplotlib
import matplotlib.pyplot as plt
import pytorch_lightning as pl
from omegaconf import DictConfig
from pytorch_lightning.callbacks import EarlyStopping, ModelCheckpoint
from pytorch_lightning.loggers import MLFlowLogger

from modrecog.data import RadioMLDataModule
from modrecog.model import ModRecogLightningModule
from modrecog.utils import data_exists, download_data, set_seed

matplotlib.use("Agg")


def _get_git_commit() -> str:
    try:
        return (
            subprocess.check_output(["git", "rev-parse", "--short", "HEAD"])
            .decode()
            .strip()
        )
    except Exception:
        return "unknown"


def _save_plots(metrics_history: dict[str, list[float]], plots_dir: Path) -> None:
    plots_dir.mkdir(parents=True, exist_ok=True)

    fig, axes = plt.subplots(1, 3, figsize=(18, 5))

    for ax, (key_train, key_val, title) in zip(
        axes,
        [
            ("train/loss", "val/loss", "Loss"),
            ("train/acc", "val/acc", "Accuracy"),
            ("val/f1", "val/f1", "Val F1"),
        ],
        strict=False,
    ):
        if key_train in metrics_history:
            ax.plot(metrics_history[key_train], label="train")
        if key_val in metrics_history:
            ax.plot(metrics_history[key_val], label="val")
        ax.set_title(title)
        ax.set_xlabel("Epoch")
        ax.legend()

    fig.tight_layout()
    out = plots_dir / "training_curves.png"
    fig.savefig(out, dpi=150)
    plt.close(fig)
    print(f"Plots saved to {out}")


class _MetricsCallback(pl.Callback):
    def __init__(self) -> None:
        self.history: dict[str, list[float]] = {}

    def on_train_epoch_end(
        self, trainer: pl.Trainer, pl_module: pl.LightningModule
    ) -> None:
        for k, v in trainer.callback_metrics.items():
            self.history.setdefault(k, []).append(float(v))


def run_training(cfg: DictConfig) -> None:
    set_seed(cfg.seed)

    if not data_exists(cfg.data.raw_path):
        print("HDF5 data file not found – attempting Kaggle download …")
        download_data(str(Path(cfg.data.raw_path).parent))

    dm = RadioMLDataModule(
        raw_path=cfg.data.raw_path,
        batch_size=cfg.data.batch_size,
        num_workers=cfg.data.num_workers,
        val_split=cfg.data.val_split,
        test_split=cfg.data.test_split,
        snr_min=cfg.data.get("snr_min"),
        snr_max=cfg.data.get("snr_max"),
        max_samples=cfg.data.get("max_samples"),
        seed=cfg.seed,
    )

    model = ModRecogLightningModule(cfg)

    mlf_logger = MLFlowLogger(
        experiment_name=cfg.logging.experiment_name,
        tracking_uri=cfg.logging.tracking_uri,
        tags={
            **dict(cfg.logging.tags),
            "git_commit": _get_git_commit(),
        },
        log_model=cfg.logging.log_model,
    )

    ckpt_dir = Path(cfg.training.checkpoint_dir)
    ckpt_dir.mkdir(parents=True, exist_ok=True)

    checkpoint_cb = ModelCheckpoint(
        dirpath=str(ckpt_dir),
        filename="best-{epoch:03d}-{val/f1:.4f}",
        monitor="val/f1",
        mode="max",
        save_top_k=1,
        save_last=True,
    )
    early_stop_cb = EarlyStopping(
        monitor="val/loss",
        patience=cfg.training.early_stopping_patience,
        mode="min",
    )
    metrics_cb = _MetricsCallback()

    trainer = pl.Trainer(
        max_epochs=cfg.training.epochs,
        accelerator=cfg.training.accelerator,
        devices=cfg.training.devices,
        precision=cfg.training.precision,
        gradient_clip_val=cfg.training.gradient_clip_val,
        accumulate_grad_batches=cfg.training.accumulate_grad_batches,
        logger=mlf_logger,
        callbacks=[checkpoint_cb, early_stop_cb, metrics_cb],
        log_every_n_steps=10,
    )

    trainer.fit(model, dm)
    trainer.test(model, dm, ckpt_path="best")

    _save_plots(metrics_cb.history, Path("plots"))
    print(f"Best checkpoint: {checkpoint_cb.best_model_path}")

    try:
        subprocess.run(["dvc", "add", str(ckpt_dir)], check=True)
        subprocess.run(["dvc", "push", "--remote", "model-store"], check=True)
        print("Model checkpoints tracked by DVC and pushed to model-store.")
    except subprocess.CalledProcessError as e:
        print(f"Warning: DVC add/push failed ({e}). Checkpoints saved locally only.")


@hydra.main(config_path="../configs", config_name="config", version_base=None)
def train(cfg: DictConfig) -> None:
    """Hydra entry-point: called when running train.py directly."""
    run_training(cfg)


if __name__ == "__main__":
    train()
