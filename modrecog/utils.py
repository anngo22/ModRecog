import random
import subprocess
from pathlib import Path

import numpy as np
import torch

MODULATION_CLASSES: list[str] = [
    "32PSK",
    "16APSK",
    "32QAM",
    "FM",
    "GMSK",
    "32APSK",
    "OQPSK",
    "8ASK",
    "BPSK",
    "8PSK",
    "AM-SSB-SC",
    "4ASK",
    "16PSK",
    "64APSK",
    "128QAM",
    "128APSK",
    "AM-DSB-SC",
    "AM-SSB-WC",
    "64QAM",
    "QPSK",
    "256QAM",
    "AM-DSB-WC",
    "OOK",
    "16QAM",
]


def allow_omegaconf_globals() -> None:
    if not hasattr(torch.serialization, "add_safe_globals"):
        return

    import functools

    _orig_load = torch.load

    @functools.wraps(_orig_load)
    def _load_compat(*args: object, **kwargs: object) -> object:
        kwargs["weights_only"] = False
        return _orig_load(*args, **kwargs)

    torch.load = _load_compat


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def download_data(dest_dir: str = "data/raw") -> None:
    dest = Path(dest_dir)
    dest.mkdir(parents=True, exist_ok=True)

    target = dest / "GOLD_XYZ_OSC.0001_1024.hdf5"
    if target.exists():
        print(f"Data already present at {target}")
        return

    print("Downloading RadioML 2018.01A from Kaggle ...")
    subprocess.run(
        [
            "kaggle",
            "datasets",
            "download",
            "-d",
            "pinxau1000/radioml2018",
            "--unzip",
            "-p",
            str(dest),
        ],
        check=True,
    )
    print(f"Dataset saved to {dest}")

    subprocess.run(["dvc", "add", str(target)], check=True)
    subprocess.run(["dvc", "push"], check=True)


def data_exists(raw_path: str) -> bool:
    return Path(raw_path).exists()
