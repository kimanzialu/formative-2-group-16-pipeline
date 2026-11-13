# Formative 2: Multimodal Identity and Product Recommendation

This repository implements the complete assignment flow:

1. Merge customer social-profile and transaction data.
2. Authenticate a user with face recognition.
3. Verify the user with a voiceprint model.
4. Recommend a product only when both authentication checks pass.

The project is intentionally simple and reproducible. It uses
scikit-learn models, saved feature tables, real image/audio samples, and a
command-line demonstration.

## Submission status

The repository includes the source data, reproducible pipelines, generated
features, trained model artifacts, evaluation outputs, notebooks, integration
demo, and the supporting [REPORT.md](REPORT.md). The only submission items
that cannot be stored in this repository are the final repository URL and any
required demonstration-video link; add those to the submission form before
submitting.

## Team responsibilities

| Member | Contribution |
| --- | --- |
| Aurele | Tabular data merge and product recommendation model |
| Frank | Image preprocessing, face features, and face model |
| Kenny | Audio preprocessing, augmentations, voice features, and voice model |
| Jacques | Multimodal integration and command-line application |

## Repository layout

```text
dataset/                         Source social and transaction CSV files
audio/                           Two phrases recorded by each speaker
data/                            Features, metrics, plots, and merged data
models/                          Saved voice and product model artifacts
face_recognition_project/        Face images, face pipeline, and face model
scripts/product_pipeline.py      Tabular merge and product model
scripts/audio_pipeline.py        Audio features, plots, and voice model
scripts/voice_verify.py          Runtime voice verification helper
integration_cli.py               Full face + voice + recommendation flow
notebooks/                        Notebook index and individual notebooks
```

## Submission contents

| Deliverable | Location |
| --- | --- |
| Written report | [REPORT.md](REPORT.md) |
| Source datasets | `dataset\`, `audio\`, and `face_recognition_project\images\` |
| Product pipeline and model | `scripts\product_pipeline.py`, `data\`, and `models\product_model.joblib` |
| Voice pipeline and model | `scripts\audio_pipeline.py`, `data\`, and `models\voice_model.joblib` |
| Face pipeline and model | `face_recognition_project\face_pipeline.py` and `face_recognition_project\` |
| Integration demonstration | `integration_cli.py` |
| Notebook index | `notebooks\README.md` |
| Face analysis notebook | `face_recognition_project\notebook.ipynb` |
| Audio analysis notebook | `notebooks\individual\03_audio_voice_model.ipynb` |

## Connected assignment workflow

Follow the assignment in this order. Each stage produces the inputs used by
the next stage and leaves its evidence in the repository:

```text
dataset/ + audio/ + face_recognition_project/images/
        |
        +-- product_pipeline.py --> data/merged_customer_data.csv
        |                           data/product_metrics.json
        |                           models/product_model.joblib
        |
        +-- audio_pipeline.py ----> data/audio_features.csv
        |                           data/voice_metrics.json
        |                           models/voice_model.joblib
        |
        +-- face_pipeline.py -----> face_recognition_project/image_features.csv
                                    face_recognition_project/face_metrics.json
                                    face_recognition_project/face_classifier.joblib
                                        |
                                        v
                                integration_cli.py
                                        |
                           face + voice agreement required
                                        |
                                        v
                           product_model recommendation
```

The notebooks explain the face and audio decisions, while the scripts provide
the reproducible command-line versions of the same work. Use
[notebooks/README.md](notebooks/README.md) to open the notebooks in the
recommended order and environment.

## Installation

From the repository root:

```powershell
python -m pip install -r requirements.txt
python -m pip install -r face_recognition_project\requirements.txt
```

## Reproduce the models

Run the tabular merge and product model:

```powershell
python scripts\product_pipeline.py
```

This creates `data\merged_customer_data.csv`, compares a majority baseline,
logistic regression, and Random Forest using a fixed stratified validation
split, then saves the best model as `models\product_model.joblib` together
with `data\product_metrics.json` and `data\product_confusion_matrix.png`.

Run the audio pipeline:

```powershell
python scripts\audio_pipeline.py
```

This reads the WAV recordings, trims silence, creates noise/pitch/time-stretch
augmentations, extracts 13 MFCCs plus roll-off, energy, and zero-crossing
rate, and saves `data\audio_features.csv`. It also saves waveform and
spectrogram evidence under `data\audio_plots\`, voice metrics under
`data\voice_metrics.json`, and `models\voice_model.joblib`. Voice verification
uses a predeclared `0.90` confidence threshold; it is not calibrated from the
final unauthorized test recordings.

Run the face pipeline from its directory:

```powershell
Set-Location face_recognition_project
python face_pipeline.py
Set-Location ..
```

It validates the real source images, creates HOG and RGB-histogram features,
applies flip and rotation augmentations, evaluates the classifier, and saves
`face_classifier.joblib`, `face_metrics.json`, `image_features.csv`, and
visual evidence. The deployment checkpoint is fitted on registered images
only; the Unauthorized folder is evaluated as a negative set, not registered.

The generated artifacts are already included for review. Re-run the pipelines
only when the source data or preprocessing changes, because retraining
replaces the saved models, metrics, tables, and plots.

## Command-line demonstration

After running the three pipelines, test a successful transaction:

```powershell
python integration_cli.py `
  face_recognition_project\images\Frank\neutral.jpeg `
  audio\frank_yes_approve_03.wav `
  --customer-id A151
```

The command runs face verification first, then voice verification, compares
the predicted identities, and reaches product recommendation only if both
checks pass for the same registered person.

Test an unauthorized face:

```powershell
python integration_cli.py `
  face_recognition_project\images\Unauthorized\neutral.jpeg `
  audio\frank_yes_approve_03.wav
```

Test an unauthorized voice:

```powershell
python integration_cli.py `
  face_recognition_project\images\Frank\neutral.jpeg `
  audio\unauthorized_yes_approve_03.wav
```

Both unauthorized paths must print `Access Denied` and return exit code `1`.
The integration rule is `face_ok and voice_ok and face_identity == voice_identity`.
You can also test the identity-mismatch branch:

```powershell
python integration_cli.py `
  face_recognition_project\images\Frank\neutral.jpeg `
  audio\kenny_yes_approve_03.wav
```

This must be rejected even when both individual models accept their inputs.

## Rubric evidence

| Rubric requirement | Evidence |
| --- | --- |
| EDA and plots | `data\product_confusion_matrix.png`, `data\audio_plots\`, and the face EDA images |
| Cleaning and merge validation | `scripts\product_pipeline.py` and `data\merged_customer_data.csv` |
| Image quantity/diversity | `face_recognition_project\images\Aurele`, `Frank`, and `Kenny`; each has neutral, smiling, and surprised images |
| Image augmentation/features | `face_recognition_project\augmented_images\` and `image_features.csv` |
| Audio quality/visualization | `audio\`, `data\audio_plots\`, and the audio notebook |
| Audio augmentation/features | `scripts\audio_pipeline.py` and `data\audio_features.csv` |
| Three trained models | `face_recognition_project\face_classifier.joblib`, `models\voice_model.joblib`, and `models\product_model.joblib` |
| Evaluation | `data\voice_metrics.json`, `data\product_metrics.json`, `face_recognition_project\face_metrics.json`, and confusion matrices |
| System simulation | `integration_cli.py` with authorized and unauthorized commands |
| Communication | [REPORT.md](REPORT.md), this README, and `notebooks\individual\03_audio_voice_model.ipynb` |

## Final submission checklist

- [x] Source datasets and generated evidence included
- [x] Three trained model artifacts included
- [x] Reproduction commands documented
- [x] Authorized and rejection paths verified
- [x] README and report included
- [ ] Add the final GitHub repository URL to the submission form
- [ ] Add the final demonstration-video link if required by the submission form

## Limitations

The supplied datasets are small educational samples, so the reported scores
are demonstration metrics rather than production security guarantees. The
saved models should be retrained whenever source images, recordings, or
tabular datasets change. Jacques has not submitted face images, so the face
model registers only Aurele, Frank, and Kenny. The `Unauthorized` folder is a
negative test set, not a registered team member, and is excluded from face
training. The audio folder does contain an unauthorized speaker for rejection
testing.
