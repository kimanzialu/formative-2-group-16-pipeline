"""Create audio evidence, features, and a reproducible voice model."""

from __future__ import annotations

import argparse
import json
import re
from pathlib import Path

import joblib
import librosa
import librosa.display
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score, log_loss
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

ROOT = Path(__file__).resolve().parents[1]
AUDIO_DIR = ROOT / "audio"
FEATURES_PATH = ROOT / "data" / "audio_features.csv"
MODEL_PATH = ROOT / "models" / "voice_model.joblib"
METRICS_PATH = ROOT / "data" / "voice_metrics.json"
PLOTS_DIR = ROOT / "data" / "audio_plots"
PATTERN = re.compile(r"^(?P<member>[a-z]+)_(?P<phrase>yes_approve|confirm_transaction)_(?P<take>\d+)\.wav$")
FEATURE_NAMES = [f"mfcc_{index}" for index in range(1, 14)] + ["rolloff", "energy", "zcr"]
VOICE_THRESHOLD = 0.90


def extract_features(y: np.ndarray, sr: int) -> dict[str, float]:
    mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=13)
    features = {f"mfcc_{index + 1}": float(mfcc[index].mean()) for index in range(13)}
    features["rolloff"] = float(librosa.feature.spectral_rolloff(y=y, sr=sr).mean())
    features["energy"] = float(librosa.feature.rms(y=y).mean())
    features["zcr"] = float(librosa.feature.zero_crossing_rate(y).mean())
    return features


def augmentations(y: np.ndarray, sr: int, seed: int) -> dict[str, np.ndarray]:
    rng = np.random.default_rng(seed)
    noise = rng.normal(0, max(float(y.std()) * 0.02, 1e-5), size=len(y))
    return {
        "original": y,
        "noise": y + noise,
        "pitch": librosa.effects.pitch_shift(y, sr=sr, n_steps=2),
        "stretch": librosa.effects.time_stretch(y, rate=1.1),
    }


def read_recordings(audio_dir: Path = AUDIO_DIR) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    for path in sorted(audio_dir.glob("*.wav")):
        match = PATTERN.match(path.name)
        if not match:
            continue
        member, phrase, take = match.group("member", "phrase", "take")
        y, sr = librosa.load(path, sr=16000)
        trimmed, _ = librosa.effects.trim(y, top_db=20)
        for augmentation, values in augmentations(trimmed, sr, seed=42).items():
            rows.append(
                {
                    "member": member,
                    "phrase": phrase,
                    "take": take,
                    "augmentation": augmentation,
                    **extract_features(values, sr),
                }
            )
    if not rows:
        raise ValueError(f"No recordings found in {audio_dir}")
    return pd.DataFrame(rows)


def save_visualizations(audio_dir: Path = AUDIO_DIR) -> None:
    PLOTS_DIR.mkdir(parents=True, exist_ok=True)
    members = ["aurele", "frank", "kenny", "unauthorized"]
    figure, axes = plt.subplots(len(members), 2, figsize=(12, 12))
    for row, member in enumerate(members):
        path = audio_dir / f"{member}_yes_approve_01.wav"
        y, sr = librosa.load(path, sr=16000)
        trimmed, _ = librosa.effects.trim(y, top_db=20)
        librosa.display.waveshow(trimmed, sr=sr, ax=axes[row, 0])
        axes[row, 0].set_title(f"{member}: waveform")
        spectrogram = librosa.amplitude_to_db(np.abs(librosa.stft(trimmed)), ref=np.max)
        librosa.display.specshow(spectrogram, sr=sr, x_axis="time", y_axis="hz", ax=axes[row, 1])
        axes[row, 1].set_title(f"{member}: spectrogram")
    figure.tight_layout()
    figure.savefig(PLOTS_DIR / "waveforms_and_spectrograms.png", dpi=150)
    plt.close(figure)


def train_voice_model(features: pd.DataFrame) -> dict[str, float]:
    authorized = features[features["member"] != "unauthorized"]
    train = authorized[authorized["take"] != "03"]
    test = authorized[(authorized["take"] == "03") & (authorized["augmentation"] == "original")]
    model = make_pipeline(StandardScaler(), LogisticRegression(max_iter=2000))
    model.fit(train[FEATURE_NAMES], train["member"])
    predictions = model.predict(test[FEATURE_NAMES])
    probabilities = model.predict_proba(test[FEATURE_NAMES])
    unauthorized = features[
        (features["member"] == "unauthorized") & (features["augmentation"] == "original")
    ]
    threshold = VOICE_THRESHOLD
    unauthorized_confidence = model.predict_proba(unauthorized[FEATURE_NAMES]).max(axis=1)
    metrics = {
        "accuracy": float(accuracy_score(test["member"], predictions)),
        "f1_macro": float(f1_score(test["member"], predictions, average="macro")),
        "log_loss": float(log_loss(test["member"], probabilities, labels=model.classes_)),
        "unauthorized_rejection_rate": float((unauthorized_confidence < threshold).mean()),
        "authorized_verification_acceptance_rate": float(
            (probabilities.max(axis=1) >= threshold).mean()
        ),
        "threshold": threshold,
        "threshold_source": "predeclared security threshold; not fit on final test recordings",
        "authorized_test_recordings": int(len(test)),
        "unauthorized_test_recordings": int(len(unauthorized)),
    }
    joblib.dump(
        {
            "model": model,
            "features": FEATURE_NAMES,
            "threshold": threshold,
            "sr": 16000,
            "top_db": 20,
            "classes": list(model.classes_),
        },
        MODEL_PATH,
    )
    METRICS_PATH.write_text(json.dumps(metrics, indent=2), encoding="utf-8")
    return metrics


def main() -> None:
    parser = argparse.ArgumentParser(description="Build audio features and voice model")
    parser.parse_args()
    features = read_recordings()
    FEATURES_PATH.parent.mkdir(parents=True, exist_ok=True)
    features.to_csv(FEATURES_PATH, index=False)
    save_visualizations()
    print("Voice metrics:", train_voice_model(features))


if __name__ == "__main__":
    main()
