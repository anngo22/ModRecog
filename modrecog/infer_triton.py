import json
from pathlib import Path

import numpy as np

from modrecog.utils import MODULATION_CLASSES


def infer_triton(
    input_file: str,
    url: str = "localhost:8000",
    model_name: str = "modrecog",
) -> list[dict]:
    import tritonclient.http as httpclient

    frames = json.loads(Path(input_file).read_text())
    data = np.array(frames, dtype=np.float32)

    client = httpclient.InferenceServerClient(url=url)

    inputs = [httpclient.InferInput("iq_samples", list(data.shape), "FP32")]
    inputs[0].set_data_from_numpy(data)

    outputs = [httpclient.InferRequestedOutput("logits")]

    result = client.infer(model_name, inputs, outputs=outputs)
    logits = result.as_numpy("logits")

    shifted = logits - logits.max(axis=1, keepdims=True)
    exp_l = np.exp(shifted)
    probs = exp_l / exp_l.sum(axis=1, keepdims=True)

    predictions = []
    for prob in probs:
        class_id = int(np.argmax(prob))
        predictions.append(
            {
                "class_id": class_id,
                "class_name": MODULATION_CLASSES[class_id],
                "confidence": float(prob[class_id]),
            }
        )

    return predictions
