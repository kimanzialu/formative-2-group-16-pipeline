# import os
from pathlib import Path

# Path to the main images folder
IMAGE_DIR = Path("images")

# Supported image formats
IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png"}

# Check whether the images folder exists
if not IMAGE_DIR.exists():
    raise FileNotFoundError("Images folder not found.")

# Inspect each member's folder
for person_folder in sorted(IMAGE_DIR.iterdir()):

    if person_folder.is_dir():

        images = [
            file for file in person_folder.iterdir()
            if file.suffix.lower() in IMAGE_EXTENSIONS
        ]

        print(f"\nPerson: {person_folder.name}")
        print(f"Number of images: {len(images)}")

        for image in sorted(images):
            print(f" - {image.name}")
