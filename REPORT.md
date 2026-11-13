# Formative 2 Report

## 1. System objective

The project implements a sequential identity and recommendation flow. A face
image is checked first, a voice recording is checked second, and a product
recommendation is shown only when both checks pass:

```text
face_ok AND voice_ok -> product recommendation
otherwise            -> Access Denied
```

The models are educational demonstrations trained on the supplied group data.
They are not production biometric-security systems.

## 2. Member contributions

- **Aurele:** cleaned and merged the social-profile and transaction datasets;
  trained the product-category recommendation model.
- **Frank:** collected and processed face images; extracted HOG/RGB features;
  trained and evaluated the face classifier.
- **Kenny:** collected the two voice phrases; created audio augmentations,
  waveform/spectrogram evidence, MFCC features, and the voice classifier.
- **Jacques:** connected the authentication checkpoints and product model in
  `integration_cli.py`.

## 3. Tabular data merge and product model

The two CSV sources are joined by normalizing the social identifier
(`A178` -> `178`) and matching it to `customer_id_legacy`. Repeated social
records are aggregated by customer using mean numeric scores and modal
categorical values. Missing transaction ratings are replaced with the training
median.

The merge produced:

- 150 transaction rows before joining
- 117 matched rows after joining
- 33 unmatched transaction rows
- 0 missing values in the model features after cleaning

The product pipeline compares a majority baseline, one-hot encoded logistic
regression, and Random Forest on the same stratified validation split. It
selects by macro-F1, then accuracy, then log loss, without using test labels
for feature construction or model selection. Its saved comparison is in
`data/product_metrics.json`; the selected-model confusion matrix is in
`data/product_confusion_matrix.png`.

The selected Random Forest achieved accuracy `0.133`, macro precision
`0.107`, macro recall `0.147`, macro-F1 `0.123`, and log loss `1.986`.
The majority baseline achieved accuracy `0.233` but macro-F1 `0.076`, so it
does not provide useful balanced recommendations. The weak predictive result
is retained as an honest limitation of the supplied data.

## 4. Face pipeline

Each source image is converted to RGB and resized to `224 x 224`. The model
combines HOG shape features with 16-bin histograms for each RGB channel.
Horizontal-flip and rotation augmentations are applied to training images.

The dataset has three registered identities (`Aurele`, `Frank`, and `Kenny`).
Each has neutral, smiling, and surprised examples. The `Unauthorized` folder
is a negative evaluation set and is excluded from registered face training.
Jacques has not submitted images, so Jacques is not claimed as a registered
face identity. The face pipeline validates readability, duplicate hashes,
class counts, and source resolutions.

Saved evidence includes:

- `face_recognition_project/image_features.csv`
- `face_recognition_project/augmented_images/`
- `face_recognition_project/face_eda.png`
- `face_recognition_project/face_augmentations.png`
- `face_recognition_project/face_confusion_matrix.png`
- `face_recognition_project/face_metrics.json`

The current registered validation metrics and unauthorized rejection rate are
written to `face_metrics.json`. The deployment checkpoint uses a predeclared
`0.98` confidence threshold; the Unauthorized images are retained as a
separate rejection evaluation and are not registered classes.

The registered validation classifier achieved accuracy `1.00`, macro-F1
`1.00`, and log loss `0.290`. The final deployment checkpoint accepted all
9 registered source images and rejected all 3 Unauthorized images. The
registered validation acceptance rate was `0.667`, showing the limitation of
threshold verification on a very small held-out set.

## 5. Audio pipeline

Each member recorded `yes approve` and `confirm transaction` phrases. The
repository contains three takes for each phrase for Aurele, Frank, Kenny, and
the unauthorized speaker. Silence is trimmed before feature extraction.

Three augmentations are generated for every recording: additive noise,
pitch-shift, and time-stretch. Each feature row contains 13 mean MFCC values,
spectral roll-off, RMS energy, and zero-crossing rate. The output contains 96
rows: 24 recordings multiplied by four augmentation states.

Saved evidence includes:

- `data/audio_features.csv`
- `data/audio_plots/waveforms_and_spectrograms.png`
- `data/voice_metrics.json`
- `models/voice_model.joblib`
- `notebooks/individual/03_audio_voice_model.ipynb`

The held-out authorized recordings are evaluated separately from threshold
selection. The deployment threshold is a predeclared `0.90` security threshold, not fit
on the final unauthorized recordings. Metrics are saved in
`data/voice_metrics.json`.

The held-out speaker classifier achieved accuracy `1.00`, macro-F1 `1.00`, and
log loss `0.034`. At the security threshold, it accepted 5/6 authorized
test recordings and rejected 6/6 unauthorized recordings.

## 6. System simulation

Train or regenerate the artifacts with:

```powershell
python scripts\product_pipeline.py
python scripts\audio_pipeline.py
Set-Location face_recognition_project
python face_pipeline.py
Set-Location ..
```

Run an authorized transaction:

```powershell
python integration_cli.py `
  face_recognition_project\images\Frank\neutral.jpeg `
  audio\frank_yes_approve_03.wav `
  --customer-id A151
```

Run unauthorized-face and unauthorized-voice demonstrations:

```powershell
python integration_cli.py `
  face_recognition_project\images\Unauthorized\neutral.jpeg `
  audio\frank_yes_approve_03.wav

python integration_cli.py `
  face_recognition_project\images\Frank\neutral.jpeg `
  audio\unauthorized_yes_approve_03.wav
```

The authorized command reaches a recommendation and exits with status `0`.
Both unauthorized commands stop before recommendation and exit with status
`1`. A Frank face paired with Kenny audio is also rejected by the identity
agreement check.

## 7. Submission checklist

- [x] Source datasets included
- [x] Tabular merge and product model included
- [x] Face images, augmentations, features, model, and evaluation included
- [x] Audio recordings, augmentations, features, plots, model, and evaluation included
- [x] Integration CLI with authorized and unauthorized branches included
- [x] README, notebook guide, and this report included
- [x] Face and audio notebooks connected to the reproducible pipelines
- [ ] Record the final CLI demonstration and add its video link to the submission
- [ ] Add the final GitHub repository URL to the submission form
