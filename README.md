## formative-2-group-16-pipeline

What we did:

1. Merging customer social-profile and transaction data.
2. Authenticating a user with face recognition.
3. Verifying the user with a voiceprint model.
4. Recommending a product only when both authentication checks pass.

The project is intentionally simple and reproducible. It uses
scikit-learn models, saved feature tables, real image/audio samples, and a
command-line demonstration.

The repository includes the source data, reproducible pipelines, generated
features, trained model artifacts, evaluation outputs, notebooks, integration
demo.

## Team responsibilities

| Member | Contribution |
| --- | --- |
| Aurele | Tabular data merge and product recommendation model |
| Frank | Image preprocessing, face features, face model, integration and command-line application |
| Kenny | Audio preprocessing, augmentations, voice features, and voice model |

## Some Submission contents

| Deliverable | Location |
| --- | --- |
| Source datasets | `dataset\`, `audio\`, and `face_recognition_project\images\` |
| Product pipeline and model | `scripts\product_pipeline.py`, `data\`, and `models\product_model.joblib` |
| Voice pipeline and model | `scripts\audio_pipeline.py`, `data\`, and `models\voice_model.joblib` |
| Face pipeline and model | `face_recognition_project\face_pipeline.py` and `face_recognition_project\` |
| Integration demonstration | `integration_cli.py` |
| Face analysis notebook | `face_recognition_project\notebook.ipynb` |
| Audio analysis notebook | `notebooks\individual\03_audio_voice_model.ipynb` |

