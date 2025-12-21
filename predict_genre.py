import librosa
import numpy as np
import joblib
import pandas as pd
import os
import yt_dlp

MODEL_PATH = "genre_model.pkl"
FEEDBACK_CSV = "feedback_data.csv"
VALID_GENRES = ["blues", "classical", "country", "disco", "hiphop", "jazz", "metal", "pop", "reggae", "rock"]

def download_audio_from_url(url):
    print(f"-Downloading audio from: {url}-")
    
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
    if not os.path.exists(file_path):
        print(f"ERROR: File '{file_path}' not found")
        return None

    try:
        total_duration = librosa.get_duration(path=file_path)

        if total_duration > 30:
            start_time = (total_duration / 2) - 15
        else:
            start_time = 0

        print(f"--- Loading 30s starting at {start_time:.1f}s (Total length: {total_duration:.1f}s) ---")

        y, sr = librosa.load(file_path, offset=start_time, duration=30)
        
    except Exception as e:
        print(f"ERROR: Could not process audio file: {e}")
        return None

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

    # 4. Tempo
    tempo, _ = librosa.beat.beat_track(y=y, sr=sr)
    features["tempo"] = float(tempo[0])

    return features

def save_feedback(features_dict, correct_genre):
 
    features_dict["genre"] = correct_genre
    
    feedback_row = pd.DataFrame([features_dict])
    
    if os.path.exists(FEEDBACK_CSV):
        feedback_row.to_csv(FEEDBACK_CSV, mode='a', header=False, index=False)
    else:
        feedback_row.to_csv(FEEDBACK_CSV, mode='w', header=True, index=False)
    
    print(f"✓ Feedback saved! ({correct_genre})")

def predict(file_path):   
    print(f"\n--- Analyzing: {file_path} ---")
    
    model = joblib.load(MODEL_PATH)
    
    features_dict = extract_features_dict(file_path)
    if features_dict is None: return

    expected_features = model.named_steps['scaler'].feature_names_in_
    X_new = pd.DataFrame([features_dict])
    X_new = X_new[expected_features]

    probabilities = model.predict_proba(X_new)[0]
    all_genres = model.classes_

    genre_prob_pairs = sorted(zip(all_genres, probabilities), key=lambda x: x[1], reverse=True)
    top_3 = genre_prob_pairs[:3]

    print("AI Top 3 Predictions:")
    for i, (genre, prob) in enumerate(top_3):
        prefix = "-> " if i == 0 else "   "
        print(f"{prefix}{genre.upper()}: {prob*100:.1f}%")

    print("-" * 30)

    feedback = input("\nWas the prediction correct? (y/n): ").strip().lower()
    
    if feedback == 'n':
        print(f"\nValid genres: {', '.join(VALID_GENRES)}")
        correct_genre = input("What's the correct genre? ").strip().lower()
        
        if correct_genre in VALID_GENRES:
            save_feedback(features_dict, correct_genre)
        else:
            print(f"Invalid genre. Must be one of: {', '.join(VALID_GENRES)}")
    elif feedback == 'y':
        print("Cool")
    else:
        print("No feedback recorded.")

if __name__ == "__main__":
    print("-Music Genre AI Agent-")
    user_input = input("Paste a YouTube URL or a local filename: ").strip()
    
    target_file = user_input
    is_temp = False

    if user_input.startswith("http"):
        try:
            target_file = download_audio_from_url(user_input)
            is_temp = True
        except Exception as e:
            print(f"Failed to download audio: {e}")
            target_file = None

    if target_file:
        predict(target_file)


        if is_temp and os.path.exists(target_file):
            os.remove(target_file)
            print("-Cleaned up temporary files-")