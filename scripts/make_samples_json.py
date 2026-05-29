import argparse
import json
from pathlib import Path

import h5py
import numpy as np

from modrecog.utils import MODULATION_CLASSES

DEFAULT_HDF5 = "data/raw/GOLD_XYZ_OSC.0001_1024.hdf5"
DEFAULT_OUT = "samples.json"


def make_samples(
    hdf5_path: str = DEFAULT_HDF5,
    n: int = 3,
    snr: int | None = None,
    out: str = DEFAULT_OUT,
) -> None:
    path = Path(hdf5_path)
    if not path.exists():
        print(f"Dataset not found at {path}. Run download-data first.")
        return

    with h5py.File(path, "r") as f:
        snr_arr = f["Z"][:].squeeze()
        labels_oh = f["Y"][:]
        labels = np.argmax(labels_oh, axis=1)

        if snr is not None:
            indices = np.where(snr_arr == snr)[0]
            if len(indices) == 0:
                print(f"No samples found at SNR={snr} dB.")
                return
            indices = indices[:n]
        else:
            indices = np.arange(min(n, len(snr_arr)))

        samples = [f["X"][int(i)].T.tolist() for i in indices]
        true_labels = [int(labels[i]) for i in indices]
        sample_snrs = [float(snr_arr[i]) for i in indices]

    Path(out).write_text(json.dumps(samples, indent=None))
    print(f"Wrote {len(samples)} sample(s) to {out}")
    for i, (lbl, db) in enumerate(zip(true_labels, sample_snrs, strict=False)):
        print(f"  [{i}] true class: {MODULATION_CLASSES[lbl]} (SNR {db:+.0f} dB)")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Create samples.json for infer")
    parser.add_argument("--hdf5_path", default=DEFAULT_HDF5)
    parser.add_argument("--n", type=int, default=3, help="Number of samples")
    parser.add_argument("--snr", type=int, default=None, help="SNR level in dB")
    parser.add_argument("--out", default=DEFAULT_OUT)
    args = parser.parse_args()
    make_samples(hdf5_path=args.hdf5_path, n=args.n, snr=args.snr, out=args.out)
