from pathlib import Path

import fire


def train(*overrides: str) -> None:
    import sys

    from hydra import compose, initialize_config_dir
    from hydra.core.global_hydra import GlobalHydra

    from modrecog.train import run_training

    if overrides:
        effective_overrides: list[str] = list(overrides)
    else:
        effective_overrides = [a for a in sys.argv[1:] if "=" in a]

    GlobalHydra.instance().clear()
    config_dir = str(Path(__file__).parent / "configs")
    with initialize_config_dir(
        config_dir=config_dir, job_name="train", version_base=None
    ):
        cfg = compose(config_name="config", overrides=effective_overrides)
    run_training(cfg)


def infer(
    input_file: str | None = None, checkpoint: str = "models/dvc/last.ckpt"
) -> None:
    import argparse

    if input_file is None:
        p = argparse.ArgumentParser(description="ModRecog inference")
        p.add_argument("--input_file", required=True, help="Path to samples JSON")
        p.add_argument("--checkpoint", default="models/dvc/last.ckpt")
        a = p.parse_args()
        input_file = a.input_file
        checkpoint = a.checkpoint

    from modrecog.infer import infer as _infer

    results = _infer(input_file, checkpoint)
    for i, r in enumerate(results):
        name = r["class_name"]
        cid = r["class_id"]
        conf = r["confidence"]
        print(f"[{i}] {name} (id={cid}, conf={conf:.4f})")


def export_onnx(
    checkpoint: str = "models/dvc/last.ckpt",
    output: str = "models/triton/modrecog/1/model.onnx",
) -> None:
    from modrecog.export import export_to_onnx

    export_to_onnx(checkpoint=checkpoint, output_path=output)


def export_tensorrt(
    onnx_path: str = "models/triton/modrecog/1/model.onnx",
    output: str = "models/triton/modrecog/1/model.trt",
    fp16: bool = True,
) -> None:
    """Convert ONNX model to TensorRT engine."""
    from modrecog.export import export_to_tensorrt

    export_to_tensorrt(onnx_path=onnx_path, output_path=output, fp16=fp16)


def download_data(dest: str = "data/raw") -> None:
    from modrecog.utils import download_data as _download

    _download(dest)


def infer_triton(
    input_file: str,
    url: str = "localhost:8000",
    model_name: str = "modrecog",
) -> None:
    """Run inference via Triton Inference Server.

    Requires Triton running (`docker compose up triton -d`) and the ONNX model
    exported to models/triton/modrecog/1/model.onnx.
    """
    from modrecog.infer_triton import infer_triton as _infer_triton

    results = _infer_triton(input_file, url=url, model_name=model_name)
    for i, r in enumerate(results):
        name = r["class_name"]
        cid = r["class_id"]
        conf = r["confidence"]
        print(f"[{i}] {name} (id={cid}, conf={conf:.4f})")


if __name__ == "__main__":
    fire.Fire(
        {
            "train": train,
            "infer": infer,
            "infer-triton": infer_triton,
            "export-onnx": export_onnx,
            "export-tensorrt": export_tensorrt,
            "download-data": download_data,
        }
    )
