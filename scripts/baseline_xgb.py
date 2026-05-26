import argparse
from pathlib import Path

import h5py
import numpy as np
from scipy import stats
from sklearn.metrics import accuracy_score, classification_report
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from xgboost import XGBClassifier

from modrecog.utils import MODULATION_CLASSES

DEFAULT_DATA = "data/raw/GOLD_XYZ_OSC.0001_1024.hdf5"
SEED = 42


def extract_features(x: np.ndarray) -> np.ndarray:
    feats: list[float] = []
    for ch in range(2):
        sig = x[:, ch].astype(np.float64)
        rms = float(np.sqrt(np.mean(sig**2)))
        peak = float(np.max(np.abs(sig)))
        feats.extend(
            [
                float(np.mean(sig)),
                float(np.std(sig)),
                float(np.var(sig)),
                float(stats.skew(sig)),
                float(stats.kurtosis(sig)),
                float(np.max(sig)),
                float(np.min(sig)),
                float(np.ptp(sig)),
                rms,
                peak / (rms + 1e-9),
                float(np.mean(np.abs(sig))),
                float(np.mean(np.diff(np.sign(sig)) != 0)),
                float(np.mean(sig**2)),
            ]
        )
    return np.array(feats, dtype=np.float32)


def load_dataset(
    hdf5_path: str,
    snr_min: int = -20,
    snr_max: int = 30,
    max_samples: int | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    print(f"Loading {hdf5_path} …")
    with h5py.File(hdf5_path, "r") as f:
        raw_x = f["X"][:]
        labels = np.argmax(f["Y"][:], axis=1).astype(np.int64)
        snr = f["Z"][:].squeeze()

    mask = (snr >= snr_min) & (snr <= snr_max)
    raw_x, labels = raw_x[mask], labels[mask]

    if max_samples is not None and len(raw_x) > max_samples:
        idx = np.random.default_rng(SEED).choice(len(raw_x), max_samples, replace=False)
        raw_x, labels = raw_x[idx], labels[idx]

    print(f"Extracting features for {len(raw_x):,} samples …")
    features = np.stack([extract_features(raw_x[i]) for i in range(len(raw_x))])
    return features, labels


def run_baseline(
    data_path: str = DEFAULT_DATA,
    snr_min: int = -20,
    snr_max: int = 30,
    max_samples: int = 50_000,
    n_estimators: int = 300,
    max_depth: int = 6,
) -> None:
    if not Path(data_path).exists():
        print(f"Data not found at {data_path}. Run download_data() first.")
        return

    X, y = load_dataset(
        data_path, snr_min=snr_min, snr_max=snr_max, max_samples=max_samples
    )

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, stratify=y, random_state=SEED
    )

    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)

    print(
        f"Train: {len(X_train):,}  Test: {len(X_test):,}  Features: {X_train.shape[1]}"
    )
    print(f"Training XGBoost (n_estimators={n_estimators}, max_depth={max_depth}) …")

    clf = XGBClassifier(
        n_estimators=n_estimators,
        max_depth=max_depth,
        learning_rate=0.1,
        subsample=0.8,
        colsample_bytree=0.8,
        use_label_encoder=False,
        eval_metric="mlogloss",
        random_state=SEED,
        n_jobs=-1,
    )
    clf.fit(X_train, y_train)

    y_pred = clf.predict(X_test)
    acc = accuracy_score(y_test, y_pred)
    print(f"\nTest accuracy: {acc:.4f} ({acc * 100:.2f}%)\n")
    print(
        classification_report(
            y_test,
            y_pred,
            target_names=MODULATION_CLASSES,
            zero_division=0,
        )
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="XGBoost baseline for ModRecog")
    parser.add_argument("--data_path", default=DEFAULT_DATA)
    parser.add_argument("--snr_min", type=int, default=-20)
    parser.add_argument("--snr_max", type=int, default=30)
    parser.add_argument("--max_samples", type=int, default=50_000)
    parser.add_argument("--n_estimators", type=int, default=300)
    parser.add_argument("--max_depth", type=int, default=6)
    args = parser.parse_args()
    run_baseline(
        data_path=args.data_path,
        snr_min=args.snr_min,
        snr_max=args.snr_max,
        max_samples=args.max_samples,
        n_estimators=args.n_estimators,
        max_depth=args.max_depth,
    )
