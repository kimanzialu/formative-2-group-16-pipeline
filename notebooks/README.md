# Notebook guide

The notebooks provide the explanatory evidence behind the reproducible
pipelines in the repository root.

## Recommended order

| Order | Notebook | Purpose | Run from |
| --- | --- | --- | --- |
| 1 | [face-recognition notebook](../face_recognition_project/notebook.ipynb) | Validates face images, demonstrates augmentation, and evaluates the face classifier | `face_recognition_project\` |
| 2 | [audio voice-model notebook](individual/03_audio_voice_model.ipynb) | Trims recordings, visualizes audio, creates augmentations, extracts features, and evaluates voice verification | Repository root or Google Colab after updating the data paths |

## How the notebooks connect to the assignment

The notebooks are explanatory companions to the scripts, not separate model
implementations:

1. Face analysis produces the evidence and model saved under
   `face_recognition_project\`.
2. Audio analysis produces the feature table, plots, metrics, and voice model
   saved under `data\` and `models\`.
3. The tabular product work is reproduced by
   `scripts\product_pipeline.py`.
4. `integration_cli.py` loads the saved face, voice, and product artifacts and
   demonstrates the complete authentication-to-recommendation flow.

For a clean submission review, read the notebooks alongside
[REPORT.md](../REPORT.md), then run the commands in the root
[README.md](../README.md).

## Execution notes

- The face notebook imports `face_pipeline.py`, so open it with
  `face_recognition_project\` as the working directory.
- The audio notebook contains Google Colab/Drive setup cells. The generated
  local artifacts in `data\` and `models\` are the portable submission outputs.
- Do not treat the unauthorized face and voice samples as registered identities;
  they are reserved for rejection testing.
