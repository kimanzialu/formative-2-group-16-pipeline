from pathlib import Path
import joblib, librosa, numpy as np, pandas as pd

MODEL_PATH = Path(__file__).resolve().parent.parent / "models" / "voice_model.joblib"

def extract_features(y, sr):
    mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=13)
    feats = {f"mfcc_{i+1}": mfcc[i].mean() for i in range(13)}
    feats["rolloff"] = librosa.feature.spectral_rolloff(y=y, sr=sr).mean()
    feats["energy"] = librosa.feature.rms(y=y).mean()
    feats["zcr"] = librosa.feature.zero_crossing_rate(y).mean()
    return feats

def verify_voice(audio_path, bundle_path=MODEL_PATH):
    """Returns (is_approved, predicted_member, confidence)."""
    bundle = joblib.load(bundle_path)
    y, sr = librosa.load(str(audio_path), sr=bundle["sr"])
    y, _ = librosa.effects.trim(y, top_db=bundle["top_db"])
    if len(y) < 0.2 * sr:
        return False, None, 0.0
    X = pd.DataFrame([extract_features(y, sr)])[bundle["features"]]
    proba = bundle["model"].predict_proba(X)[0]
    i = int(np.argmax(proba))
    member, conf = bundle["classes"][i], float(proba[i])
    ok = conf >= bundle["threshold"]
    return bool(ok), (member if ok else None), conf