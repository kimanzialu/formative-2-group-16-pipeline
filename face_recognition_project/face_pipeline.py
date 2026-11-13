"""Simple, reproducible face preprocessing and authentication pipeline."""

from __future__ import annotations

import argparse
import hashlib
import json
import random
from pathlib import Path

import joblib
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image, ImageEnhance, ImageOps
from skimage.color import rgb2gray
from skimage.feature import hog
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    ConfusionMatrixDisplay,
    accuracy_score,
    f1_score,
    log_loss,
)
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

SEED = 42
IMAGE_SIZE = (224, 224)
IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png"}
REGISTERED_LABELS = ("Aurele", "Frank", "Kenny")
VERIFICATION_THRESHOLD = 0.98


def is_readable_image(path: Path) -> bool:
    if path.suffix.lower() in IMAGE_EXTENSIONS:
        return True
    try:
        with Image.open(path):
            return True
    except (OSError, ValueError):
        return False


def patch_dataset(image_root: Path) -> None:
    if not image_root.is_dir():
        raise FileNotFoundError(f"Missing dataset directory: {image_root}")
    print("Using real images only; no identity images are generated.")


def image_paths(image_root: Path) -> list[tuple[Path, str]]:
    paths = []
    for folder in sorted(image_root.iterdir()):
        label = folder.name
        if folder.exists():
            paths.extend(
                (path, label)
                for path in sorted(folder.iterdir())
                if path.is_file() and is_readable_image(path)
            )
    return paths


def registered_image_paths(image_root: Path) -> list[tuple[Path, str]]:
    return [
        (path, label)
        for path, label in image_paths(image_root)
        if label in REGISTERED_LABELS
    ]


def validate_dataset(image_root: Path) -> dict[str, int]:
    seen_hashes: set[str] = set()
    duplicate_count = 0
    dimensions: list[tuple[int, int]] = []
    paths = image_paths(image_root)
    for path, _ in paths:
        try:
            with Image.open(path) as image:
                image.verify()
            with Image.open(path) as image:
                dimensions.append(image.size)
                digest = hashlib.sha256(path.read_bytes()).hexdigest()
                if digest in seen_hashes:
                    duplicate_count += 1
                seen_hashes.add(digest)
        except Exception as error:
            raise ValueError(f"Invalid image {path}: {error}") from error
    counts = {
        label: sum(item_label == label for _, item_label in paths)
        for label in sorted({item_label for _, item_label in paths})
    }
    print(
        f"Validated {len(paths)} readable images; duplicates: {
            duplicate_count}")
    print(f"Resolution range: {min(dimensions)} to {max(dimensions)}")
    print("Class counts:", counts)
    return counts


def run_eda(image_root: Path, output_path: Path) -> None:
    paths = image_paths(image_root)
    labels = [label for _, label in paths]
    sizes = [Image.open(path).size for path, _ in paths]
    class_names = sorted(set(labels))
    counts = [labels.count(label) for label in class_names]
    figure, axes = plt.subplots(1, 3, figsize=(16, 4))
    axes[0].bar(class_names, counts, color="steelblue")
    axes[0].set_title("Image count by identity")
    axes[0].tick_params(axis="x", rotation=45)
    axes[1].scatter(
        [width for width, _ in sizes], [height for _, height in sizes],
        c="darkorange")
    axes[1].set_title("Source image dimensions")
    axes[1].set_xlabel("Width")
    axes[1].set_ylabel("Height")
    axes[2].bar(
        ["Minimum width", "Maximum width"],
        [min(width for width, _ in sizes), max(width for width, _ in sizes)])
    axes[2].set_title("Resolution range confirms resize is needed")
    figure.tight_layout()
    figure.savefig(output_path, dpi=150)
    plt.show()


def augment(image: Image.Image, name: str) -> Image.Image:
    if name == "horizontal_flip":
        return ImageOps.mirror(image)
    if name == "rotation":
        return image.rotate(12, resample=Image.Resampling.BILINEAR)
    if name == "brightness":
        return ImageEnhance.Brightness(image).enhance(1.25)
    raise ValueError(f"Unknown augmentation: {name}")


def show_augmentations(
    paths: list[tuple[Path, str]], output_path: Path, augmented_root: Path
) -> None:
    for path, label in paths:
        with Image.open(path) as source:
            image = source.convert("RGB").resize(IMAGE_SIZE)
        for name in ("horizontal_flip", "rotation", "brightness"):
            target = augmented_root / label / f"{path.stem}__{name}.jpeg"
            target.parent.mkdir(parents=True, exist_ok=True)
            augment(image, name).save(target, quality=92)

    path, label = paths[0]
    with Image.open(path) as source:
        image = source.convert("RGB").resize(IMAGE_SIZE)
    variants = [("original", image)] + [
        (name, augment(image, name)) for name in ("horizontal_flip",
                                                  "rotation", "brightness")
    ]
    figure, axes = plt.subplots(1, 4, figsize=(14, 4))
    for axis, (name, variant) in zip(axes, variants):
        axis.imshow(variant)
        axis.set_title(f"{label}: {name}")
        axis.axis("off")
    figure.tight_layout()
    figure.savefig(output_path, dpi=150)
    plt.show()


def extract_features(image: Image.Image) -> np.ndarray:
    array = np.asarray(image.convert("RGB").resize(IMAGE_SIZE),
                       dtype=np.float32) / 255.0
    gray = rgb2gray(array)
    edges = hog(
        gray,
        orientations=9,
        pixels_per_cell=(16, 16),
        cells_per_block=(2, 2),
        block_norm="L2-Hys",
    )
    histograms = [
        np.histogram(array[:, :, channel], bins=16,
                     range=(0, 1), density=True)[0]
        for channel in range(3)
    ]
    return np.concatenate([edges, *histograms]).astype(np.float32)


def feature_row(path: Path, label: str,
                augmentation: str = "original") -> tuple:
    with Image.open(path) as source:
        image = source.convert("RGB").resize(IMAGE_SIZE)
    if augmentation != "original":
        image = augment(image, augmentation)
    return path.name, augmentation, label, extract_features(image)


def train_and_evaluate(image_root: Path, output_dir: Path) -> dict:
    all_paths = image_paths(image_root)
    registered_paths = registered_image_paths(image_root)
    original_rows = [feature_row(path, label) for path, label in registered_paths]
    unauthorized_rows = [
        feature_row(path, label)
        for path, label in all_paths
        if label == "Unauthorized"
    ]
    if len(set(label for _, _, label, _ in original_rows)) < 2:
        raise ValueError("At least two image classes are required")
    original_indices = np.arange(len(original_rows))
    labels = np.array([row[2] for row in original_rows])
    class_count = len(np.unique(labels))
    validation_size = max(class_count, int(np.ceil(0.25 * len(original_rows))))
    train_indices, validation_indices = train_test_split(
        original_indices,
        test_size=validation_size,
        random_state=SEED,
        stratify=labels,
    )
    augmented_rows = []
    for index in train_indices:
        path, label = image_paths(image_root)[index]
        for name in ("horizontal_flip", "rotation"):
            augmented_rows.append(feature_row(path, label, name))
    csv_rows = original_rows + unauthorized_rows + augmented_rows
    header = ["image_name", "augmentation", "label"] + [
        f"feature_{index:04d}" for index in range(len(csv_rows[0][3]))
    ]
    with (output_dir / "image_features.csv").open("w",
                                                  encoding="utf-8",
                                                  newline="") as file:
        file.write(",".join(header) + "\n")
        for image_name, augmentation_name, label, features in csv_rows:
            values = [image_name, augmentation_name, label]
            values.extend(f"{value:.8g}" for value in features)
            file.write(",".join(values) + "\n")
    training_rows = [
        original_rows[index] for index in train_indices] + augmented_rows
    validation_rows = [original_rows[index] for index in validation_indices]
    training_x = np.vstack([row[3] for row in training_rows])
    training_y = np.array([row[2] for row in training_rows])
    validation_x = np.vstack([row[3] for row in validation_rows])
    validation_y = np.array([row[2] for row in validation_rows])
    model = Pipeline(
        [
            ("scale", StandardScaler()),
            ("classifier", LogisticRegression(max_iter=2000,
                                              class_weight="balanced")),
        ]
    )
    model.fit(training_x, training_y)
    predictions = model.predict(validation_x)
    probabilities = model.predict_proba(validation_x)
    metrics = {
        "accuracy": float(accuracy_score(validation_y, predictions)),
        "f1_macro": float(
            f1_score(validation_y, predictions, average="macro",
                     zero_division=0)),
        "log_loss": float(log_loss(validation_y, probabilities,
                                   labels=model.classes_)),
        "verification_threshold": VERIFICATION_THRESHOLD,
        "registered_validation_acceptance_rate": float(
            (probabilities.max(axis=1) >= VERIFICATION_THRESHOLD).mean()
        ),
        "registered_image_count": len(registered_paths),
        "unauthorized_image_count": len(unauthorized_rows),
    }
    print("Validation metrics:", metrics)
    figure, axis = plt.subplots(figsize=(7, 6))
    ConfusionMatrixDisplay.from_predictions(
        validation_y, predictions, labels=model.classes_, xticks_rotation=45,
        ax=axis)
    figure.tight_layout()
    figure.savefig(output_dir / "face_confusion_matrix.png", dpi=150)
    deployment_rows = list(original_rows)
    for path, label in registered_paths:
        for name in ("horizontal_flip", "rotation"):
            deployment_rows.append(feature_row(path, label, name))
    deployment_model = Pipeline(
        [
            ("scale", StandardScaler()),
            ("classifier", LogisticRegression(max_iter=2000,
                                              class_weight="balanced")),
        ]
    )
    deployment_model.fit(
        np.vstack([row[3] for row in deployment_rows]),
        np.array([row[2] for row in deployment_rows]),
    )
    registered_deployment_confidence = deployment_model.predict_proba(
        np.vstack([row[3] for row in original_rows])
    ).max(axis=1)
    metrics["registered_deployment_acceptance_rate"] = float(
        (registered_deployment_confidence >= VERIFICATION_THRESHOLD).mean()
    )
    unauthorized_x = np.vstack([row[3] for row in unauthorized_rows])
    unauthorized_predictions = deployment_model.predict(unauthorized_x)
    unauthorized_confidence = deployment_model.predict_proba(unauthorized_x).max(axis=1)
    metrics["unauthorized_rejection_rate"] = float(
        (unauthorized_confidence < VERIFICATION_THRESHOLD).mean()
    )
    metrics["unauthorized_predictions"] = unauthorized_predictions.tolist()
    (output_dir / "face_metrics.json").write_text(
        json.dumps(metrics, indent=2), encoding="utf-8"
    )
    joblib.dump(
        {
            "model": deployment_model,
            "labels": list(deployment_model.classes_),
            "threshold": VERIFICATION_THRESHOLD,
            "registered_labels": list(REGISTERED_LABELS),
        },
        output_dir / "face_classifier.joblib",
    )
    return metrics


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Build and evaluate the face model")
    parser.add_argument("--root", type=Path, default=Path.cwd())
    args = parser.parse_args()
    random.seed(SEED)
    np.random.seed(SEED)
    output_dir = args.root
    image_root = output_dir / "images"
    patch_dataset(image_root)
    validate_dataset(image_root)
    run_eda(image_root, output_dir / "face_eda.png")
    paths = registered_image_paths(image_root)
    show_augmentations(paths, output_dir / "face_augmentations.png",
                       output_dir / "augmented_images")
    metrics = train_and_evaluate(image_root, output_dir)
    print(f"Saved face_classifier.joblib and visual evidence in {output_dir}")
    print(metrics)


if __name__ == "__main__":
    main()
