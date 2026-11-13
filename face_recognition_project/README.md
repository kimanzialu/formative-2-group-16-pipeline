# Face recognition component

This folder contains the image-authentication component of the multimodal
Formative 2 pipeline.

## Workflow

1. Convert source images to RGB and resize them to `224 x 224`.
2. Extract HOG shape features and 16-bin RGB histograms.
3. Generate horizontal-flip and rotation augmentations for training.
4. Train and evaluate the balanced logistic-regression classifier.
5. Apply the saved confidence threshold in the authentication CLI.

## Run from this directory

Install the face-specific dependencies:

```powershell
python -m pip install -r requirements.txt
```

Train the face model and generate its evidence:

```powershell
python face_pipeline.py
```

Authenticate a registered image:

```powershell
python face_auth_cli.py images\Frank\neutral.jpeg
```

The command exits with status `0` for an accepted registered identity and
status `1` for a rejected or unauthorized image.

## Outputs

The pipeline writes the face model, feature table, metrics, confusion matrix,
EDA plot, augmentation examples, and generated images to this directory.
`images\Unauthorized\` is a negative evaluation set; it is not registered as
a class.

The face model is consumed by the root-level `integration_cli.py`, together
with the voice and product models. See the root [README.md](../README.md) for
the complete submission workflow.
