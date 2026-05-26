import json
from pathlib import Path
from typing import Any

import numpy as np
import torch

from modrecog.utils import MODULATION_CLASSES, allow_omegaconf_globals


def infer(input_file: str, checkpoint: str) -> list[dict[str, Any]]:
    from modrecog.model import ModRecogLightningModule

    allow_omegaconf_globals()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model = ModRecogLightningModule.load_from_checkpoint(
        checkpoint, map_location=device
    )
    model.eval()
    model.to(device)

    with Path(input_file).open() as f:
        samples = json.load(f)

    results: list[dict[str, Any]] = []
    with torch.no_grad():
        for sample in samples:
            x = (
                torch.tensor(np.array(sample), dtype=torch.float32)
                .unsqueeze(0)
                .to(device)
            )
            logits = model(x)
            probs = torch.softmax(logits, dim=1).squeeze(0)
            class_id = int(probs.argmax().item())
            results.append(
                {
                    "class_id": class_id,
                    "class_name": MODULATION_CLASSES[class_id],
                    "confidence": float(probs[class_id].item()),
                }
            )

    return results


if __name__ == "__main__":
    infer("input.json", "models/dvc/last.ckpt")
