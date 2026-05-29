from pathlib import Path

import torch

from modrecog.utils import allow_omegaconf_globals


def export_to_onnx(
    checkpoint: str,
    output_path: str = "models/triton/modrecog/1/model.onnx",
    input_channels: int = 2,
    window_length: int = 1024,
    opset: int = 17,
) -> None:
    from modrecog.model import ModRecogLightningModule

    allow_omegaconf_globals()

    out = Path(output_path)
    out.parent.mkdir(parents=True, exist_ok=True)

    model = ModRecogLightningModule.load_from_checkpoint(checkpoint, map_location="cpu")
    model.eval()

    dummy = torch.randn(1, input_channels, window_length)
    torch.onnx.export(
        model,
        dummy,
        str(out),
        input_names=["iq_samples"],
        output_names=["logits"],
        dynamic_axes={"iq_samples": {0: "batch_size"}, "logits": {0: "batch_size"}},
        opset_version=opset,
    )
    print(f"ONNX model saved to {out}")


if __name__ == "__main__":
    export_to_onnx("models/dvc/last.ckpt")
