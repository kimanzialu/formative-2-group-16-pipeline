"""Run the face + voice authentication flow before product recommendation."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

from face_recognition_project.face_auth_cli import verify_face
from scripts.product_pipeline import recommend_product
from scripts.voice_verify import verify_voice

ROOT = Path(__file__).resolve().parent
FACE_CHECKPOINT = ROOT / "face_recognition_project" / "face_classifier.joblib"


def authenticate_face(image_path: Path) -> tuple[bool, str, float]:
    return verify_face(image_path, FACE_CHECKPOINT)


def main() -> int:
    parser = argparse.ArgumentParser(description="Multimodal identity and recommendation demo")
    parser.add_argument("image_path", type=Path)
    parser.add_argument("audio_path", type=Path)
    parser.add_argument("--customer-id", default="A151")
    args = parser.parse_args()

    try:
        face_ok, face_member, face_confidence = authenticate_face(args.image_path)
        print(
            f"Face prediction: {face_member or 'unverified'} "
            f"({face_confidence:.1%}); "
            f"verification: {'ACCEPT' if face_ok else 'REJECT'}"
        )
        if not face_ok:
            print("Access Denied: face verification failed.")
            return 1

        voice_ok, voice_member, voice_confidence = verify_voice(args.audio_path)
        print(
            f"Voice prediction: {voice_member or 'unverified'} "
            f"({voice_confidence:.1%}); "
            f"verification: {'ACCEPT' if voice_ok else 'REJECT'}"
        )
        if not voice_ok:
            print("Access Denied: voice verification failed.")
            return 1
        if face_member.casefold() != voice_member.casefold():
            print(
                f"Access Denied: identity mismatch "
                f"(face={face_member}, voice={voice_member})."
            )
            return 1

    except (FileNotFoundError, OSError, ValueError, KeyError) as error:
        print(f"Authentication unavailable: {error}", file=sys.stderr)
        return 2
    try:
        product, product_confidence = recommend_product(args.customer_id)
    except (FileNotFoundError, OSError, ValueError, KeyError) as error:
        print(f"Recommendation unavailable: {error}", file=sys.stderr)
        return 2
    print(
        f"Access Granted. Recommended product: {product} "
        f"({product_confidence:.1%} confidence)."
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
