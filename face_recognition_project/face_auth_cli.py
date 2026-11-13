import argparse
from pathlib import Path

import joblib
import numpy as np
from PIL import Image
from skimage.color import rgb2gray
from skimage.feature import hog

DEFAULT_CHECKPOINT = Path(__file__).with_name("face_classifier.joblib")


def extract_features(path: Path) -> np.ndarray:
    image = Image.open(path).convert("RGB").resize((224, 224))
    array = np.asarray(image, dtype=np.float32) / 255.0
    gray = rgb2gray(array)
    hog_features = hog(
        gray,
        orientations=9,
        pixels_per_cell=(16, 16),
        cells_per_block=(2, 2),
        block_norm="L2-Hys",
    )
    histograms = [
        np.histogram(
            array[:, :, channel], bins=16, range=(0, 1), density=True
        )[0]
        for channel in range(3)
    ]
    return np.concatenate([hog_features, *histograms]).astype(np.float32)


def verify_face(
    image_path: str | Path,
    checkpoint_path: str | Path = DEFAULT_CHECKPOINT,
) -> tuple[bool, str, float]:
    """Return (verified, predicted_identity, confidence)."""
    checkpoint = joblib.load(checkpoint_path)
    path = Path(image_path)
    if not path.is_file():
        raise FileNotFoundError(f"Image not found: {path}")
    probabilities = checkpoint["model"].predict_proba([extract_features(path)])[0]
    index = int(np.argmax(probabilities))
    label = str(checkpoint["model"].classes_[index])
    confidence = float(probabilities[index])
    verified = (
        label in checkpoint["registered_labels"]
        and confidence >= checkpoint["threshold"]
    )
    return verified, label, confidence


def main(image_path: str, checkpoint_path: str | Path = DEFAULT_CHECKPOINT) -> bool:
    verified, identity, confidence = verify_face(image_path, checkpoint_path)
    predicted = identity or "unverified"
    status = "ACCEPT" if verified else "REJECT"
    print(
        f"Predicted identity: {predicted}; confidence: {confidence:.1%}; "
        f"verification: {status}"
    )
    return verified


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Facial recognition security checkpoint")
    parser.add_argument("image_path", nargs="?", help="Image to classify")
    parser.add_argument("--checkpoint", default=str(DEFAULT_CHECKPOINT))
    args = parser.parse_args()
    supplied = args.image_path or input("Image path: ").strip()
    raise SystemExit(0 if main(supplied, args.checkpoint) else 1)
