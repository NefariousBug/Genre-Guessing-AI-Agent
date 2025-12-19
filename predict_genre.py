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

    total_duration = librosa.get_duration(path=file_path)

    if total_duration > 30:
        start_time = (total_duration / 2) - 15
    else:
        start_time = 0

    print(f"--- Loading 30s starting at {start_time:.1f}s (Total length: {total_duration:.1f}s) ---")

    y, sr = librosa.load(file_path, offset=start_time, duration=30)
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
    tempo, _ = librosa.beat.beat_track(y=y, sr=sr)
    features["tempo"] = float(tempo[0])

    return features

def save_feedback(features_dict, correct_genre):
    """Saves user feedback to the feedback CSV file."""
    # Add the genre to the features
    features_dict["genre"] = correct_genre
    
    # Convert to DataFrame
    feedback_row = pd.DataFrame([features_dict])
    
    # Append to CSV (create if doesn't exist)
    if os.path.exists(FEEDBACK_CSV):
        feedback_row.to_csv(FEEDBACK_CSV, mode='a', header=False, index=False)
    else:
        feedback_row.to_csv(FEEDBACK_CSV, mode='w', header=True, index=False)
    
    print(f"✓ Feedback saved! ({correct_genre})")

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

    probabilities = model.predict_proba(X_new)[0]
    all_genres = model.classes_

    # 5. Sort them to find the Top 3
    # zip combines genre names and their scores, sorted() puts the highest first
    genre_prob_pairs = sorted(zip(all_genres, probabilities), key=lambda x: x[1], reverse=True)
    top_3 = genre_prob_pairs[:3]

    # 6. Display results
    print("AI Top 3 Predictions:")
    for i, (genre, prob) in enumerate(top_3):
        prefix = "-> " if i == 0 else "   " # Add a little arrow to the winner
        print(f"{prefix}{genre.upper()}: {prob*100:.1f}%")

    print("-" * 30)

    # 7. Collect feedback
    feedback = input("\nWas the prediction correct? (y/n): ").strip().lower()
    
    if feedback == 'n':
        print(f"\nValid genres: {', '.join(VALID_GENRES)}")
        correct_genre = input("What's the correct genre? ").strip().lower()
        
        if correct_genre in VALID_GENRES:
            save_feedback(features_dict, correct_genre)
        else:
            print(f"❌ Invalid genre. Must be one of: {', '.join(VALID_GENRES)}")
    elif feedback == 'y':
        print("Awesome!")
    else:
        print("No feedback recorded.")

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