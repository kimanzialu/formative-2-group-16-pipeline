# Face Recognition

This folder contains the image-authentication component for the Formative 2
pipeline. It predicts a person’s identity from a face image and returns an
access decision for the group’s multimodal authentication flow.

The implementation uses a small, explainable machine-learning pipeline rather
than a deep-learning model:

1. Convert each image to RGB and resize it to `224 x 224`.
2. Extract HOG features for facial edges and shape.
3. Extract 16-bin histograms for each RGB channel.
4. Augment training images with horizontal flips and 12-degree rotations.
5. Scale the features and train a balanced logistic-regression classifier.
6. Use the trained checkpoint to grant or deny access.

## Project layout

```text
face_recognition_project/
├── images/                  # Real source images, grouped by identity
├── augmented_images/        # Generated training augmentations
├── face_pipeline.py         # Validation, feature extraction, training, and evaluation
├── face_auth_cli.py         # Command-line authentication checkpoint
├── notebook.ipynb           # Interactive version of the workflow
├── requirements.txt         # Python dependencies
└── README.md
```

The current dataset contains three identities: `Aurele`, `Frank`, and `Kenny`.
Each identity currently has `neutral`, `smiling`, and `surprised` images.
Only real source images are used; generated augmentations are kept outside the
source dataset.

## Setup

From this directory, create or activate a Python environment and install the
dependencies:

```powershell
python -m pip install -r requirements.txt
```

## Train the model

Run the complete pipeline from `face_recognition_project`:

```powershell
python face_pipeline.py
```

The pipeline validates readable images, reports class counts and source
resolutions, creates training features, evaluates the classifier, and saves
the model checkpoint.

The validation output includes:

- Accuracy
- Macro-F1
- Multiclass log loss
- A confusion matrix

## Authenticate an image

Use the generated checkpoint with an image path:

```powershell
python face_auth_cli.py images\Frank\neutral.jpeg
python face_auth_cli.py images\Kenny\smiling.jpeg
```

Successful predictions print the recognized identity and exit with status
`0`. Denied or unrecognized predictions print an access-denied message and
exit with status `1`.

The checkpoint accepts the registered identities learned from the dataset.
The CLI also supports `Jacques` as a future registered identity, but a
`Jacques` class must be added to `images\Jacques\` and the model retrained
before it can be predicted. An optional checkpoint path can be supplied with:

```powershell
python face_auth_cli.py path\to\image.jpeg --checkpoint path\to\face_classifier.joblib
```

## Generated files

Running `face_pipeline.py` creates:

| File or directory | Purpose |
| --- | --- |
| `face_classifier.joblib` | Trained model used by the CLI |
| `image_features.csv` | Original and augmented feature rows |
| `face_eda.png` | Class-count and image-resolution visualizations |
| `face_augmentations.png` | Examples of the generated transformations |
| `face_confusion_matrix.png` | Validation confusion matrix |
| `augmented_images/` | Saved flip, rotation, and brightness examples |

## Multimodal integration

The face checkpoint can be combined with the voice checkpoint before allowing
the product-recommendation stage to continue:

```python
face_ok = face_authenticate(image_path)
voice_ok = voice_authenticate(audio_path)

if face_ok and voice_ok:
    run_product_recommendation()
else:
    print("Access Denied")
```

This component is intentionally limited to face-based authentication. The
tabular merge, voice model, and product recommender remain separate parts of
the group pipeline.
