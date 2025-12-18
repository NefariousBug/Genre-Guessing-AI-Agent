import librosa
import numpy as np
import joblib
import pandas as pd

FEATURE_COLUMNS = [
    # MFCCs
    *[f"mfcc_{i}_mean" for i in range(13)],
    *[f"mfcc_{i}_var" for i in range(13)],

    # Chroma
    *[f"chroma_{i}_mean" for i in range(12)],
    *[f"chroma_{i}_var" for i in range(12)],

    # Spectral
    "spectral_centroid",
    "spectral_bandwidth",
    "spectral_rolloff",
    "zero_crossing_rate",

    # Rhythm
    "tempo"
]

MODEL_PATH = "genre_model.pkl"

def extract_features(file_path):
    import os
    if not os.path.exists(file_path):
        print(f"ERROR: File '{file_path}' not found!")
        return None

    y, sr = librosa.load(file_path, duration=30)

    features = {}

    # MFCCs
    mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=13)
    for i in range(13):
        features[f"mfcc_{i}_mean"] = np.mean(mfcc[i])
        features[f"mfcc_{i}_var"] = np.var(mfcc[i])

    # Chroma
    chroma = librosa.feature.chroma_stft(y=y, sr=sr)
    for i in range(12):
        features[f"chroma_{i}_mean"] = np.mean(chroma[i])
        features[f"chroma_{i}_var"] = np.var(chroma[i])

    # Spectral features
    features["spectral_centroid"] = np.mean(librosa.feature.spectral_centroid(y=y, sr=sr))
    features["spectral_bandwidth"] = np.mean(librosa.feature.spectral_bandwidth(y=y, sr=sr))
    features["spectral_rolloff"] = np.mean(librosa.feature.spectral_rolloff(y=y, sr=sr))
    features["zero_crossing_rate"] = np.mean(librosa.feature.zero_crossing_rate(y))

    # Tempo
    features["tempo"] = float(librosa.beat.tempo(y=y, sr=sr)[0])
    
    feature_vector = [features[col] for col in FEATURE_COLUMNS]
    return np.array(feature_vector).reshape(1, -1)

def predict(file_path):   
    print(f"\n--- Analyzing: {file_path} ---")
    
    # 1. Load the model
    model = joblib.load(MODEL_PATH)
    
    # 2. Get the EXACT feature names the model expects
    # Since it's a Pipeline, the names are stored in the first step (scaler)
    expected_features = model.named_steps['scaler'].feature_names_in_
    
    # 3. Extract your raw features as a dictionary first
    # (We need to modify the end of extract_features slightly, or just do this:)
    y, sr = librosa.load(file_path, duration=30)
    
    # --- Feature Extraction Logic ---
    features = {}
    mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=13)
    for i in range(13):
        features[f"mfcc_{i}_mean"] = np.mean(mfcc[i])
        features[f"mfcc_{i}_var"] = np.var(mfcc[i])
    
    chroma = librosa.feature.chroma_stft(y=y, sr=sr)
    for i in range(12):
        features[f"chroma_{i}_mean"] = np.mean(chroma[i])
        features[f"chroma_{i}_var"] = np.var(chroma[i])

    features["spectral_centroid"] = np.mean(librosa.feature.spectral_centroid(y=y, sr=sr))
    features["spectral_bandwidth"] = np.mean(librosa.feature.spectral_bandwidth(y=y, sr=sr))
    features["spectral_rolloff"] = np.mean(librosa.feature.spectral_rolloff(y=y, sr=sr))
    features["zero_crossing_rate"] = np.mean(librosa.feature.zero_crossing_rate(y))
    features["tempo"] = float(librosa.beat.tempo(y=y, sr=sr)[0])
    
    # 4. Create the DataFrame and FORCE the order to match the model
    X_new = pd.DataFrame([features])
    X_new = X_new[expected_features] # This re-orders everything perfectly

    # 5. Final Prediction
    prediction = model.predict(X_new)[0]
    probabilities = model.predict_proba(X_new)
    confidence = np.max(probabilities)

    print(f"Predicted genre: {prediction}")
    print(f"Confidence: {confidence:.2f}")

if __name__ == "__main__":
    predict("test_song.wav")  # replace with your song
