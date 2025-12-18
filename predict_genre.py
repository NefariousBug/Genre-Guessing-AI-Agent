import librosa
import numpy as np
import joblib
import pandas as pd
import os
import yt_dlp

MODEL_PATH = "genre_model.pkl"

def download_audio_from_url(url):
    """Downloads audio from a URL and saves it as a temporary wav file."""
    print(f"--- Downloading audio from: {url} ---")
    
    ydl_opts = {
        'format': 'bestaudio/best',
        'outtmpl': 'temp_download.%(ext)s',
        'postprocessors': [{
            'key': 'FFmpegExtractAudio',
            'preferredcodec': 'wav',
            'preferredquality': '192',
        }],
        'quiet': True,
        'no_warnings': True,
    }

    with yt_dlp.YoutubeDL(ydl_opts) as ydl:
        ydl.download([url])
    
    return "temp_download.wav"

def extract_features_dict(file_path):
    """Extracts features and returns them as a dictionary."""
    if not os.path.exists(file_path):
        print(f"ERROR: File '{file_path}' not found!")
        return None

    # Load 30 seconds of audio
    y, sr = librosa.load(file_path, duration=30)
    features = {}

    # 1. MFCCs
    mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=13)
    for i in range(13):
        features[f"mfcc_{i}_mean"] = np.mean(mfcc[i])
        features[f"mfcc_{i}_var"] = np.var(mfcc[i])

    # 2. Chroma
    chroma = librosa.feature.chroma_stft(y=y, sr=sr)
    for i in range(12):
        features[f"chroma_{i}_mean"] = np.mean(chroma[i])
        features[f"chroma_{i}_var"] = np.var(chroma[i])

    # 3. Spectral Features
    features["spectral_centroid"] = np.mean(librosa.feature.spectral_centroid(y=y, sr=sr))
    features["spectral_bandwidth"] = np.mean(librosa.feature.spectral_bandwidth(y=y, sr=sr))
    features["spectral_rolloff"] = np.mean(librosa.feature.spectral_rolloff(y=y, sr=sr))
    features["zero_crossing_rate"] = np.mean(librosa.feature.zero_crossing_rate(y))

    # 4. Tempo (using updated path to avoid warnings)
    features["tempo"] = float(librosa.beat.tempo(y=y, sr=sr)[0])

    return features

def predict(file_path):   
    print(f"\n--- Analyzing: {file_path} ---")
    
    # 1. Load the model
    model = joblib.load(MODEL_PATH)
    
    # 2. Extract features
    features_dict = extract_features_dict(file_path)
    if features_dict is None: return

    # 3. Align features with model expectations
    expected_features = model.named_steps['scaler'].feature_names_in_
    X_new = pd.DataFrame([features_dict])
    X_new = X_new[expected_features]

    # 4. Final Prediction
    prediction = model.predict(X_new)[0]
    probabilities = model.predict_proba(X_new)
    confidence = np.max(probabilities)

    print(f"Predicted genre: {prediction.upper()}")
    print(f"Confidence: {confidence:.2f}")

if __name__ == "__main__":
    print("=== Music Genre AI Agent ===")
    user_input = input("Paste a YouTube URL or a local filename (e.g., song.wav): ").strip()
    
    target_file = user_input
    is_temp = False

    # Check if input is a URL
    if user_input.startswith("http"):
        try:
            target_file = download_audio_from_url(user_input)
            is_temp = True
        except Exception as e:
            print(f"Failed to download audio: {e}")
            target_file = None

    if target_file:
        predict(target_file)

        # Clean up temp file if it was downloaded
        if is_temp and os.path.exists(target_file):
            os.remove(target_file)
            print("--- Cleaned up temporary files ---")