import os
import librosa
import numpy as np
import pandas as pd

DATASET_PATH = "data/gtzan"
OUTPUT_CSV = "gtzan_features.csv"

def extract_features(file_path):
    try:
        y, sr = librosa.load(file_path, duration=30)
    except Exception as e:
        print(f"Skipping file {file_path}: {e}")
        return None

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
    features["tempo"] = librosa.beat.tempo(y=y, sr=sr)[0]

    return features


def build_dataset():
    data = []

    for genre in os.listdir(DATASET_PATH):
        genre_path = os.path.join(DATASET_PATH, genre)

        if not os.path.isdir(genre_path):
            continue

        for file in os.listdir(genre_path):
            if not file.endswith(".wav"):
                continue

            file_path = os.path.join(genre_path, file)
            print(f"Processing {file_path}")

            features = extract_features(file_path)
            if features is None:
                continue
            features["genre"] = genre
            data.append(features)

    df = pd.DataFrame(data)
    df.to_csv(OUTPUT_CSV, index=False)
    print(f"\nSaved dataset to {OUTPUT_CSV}")


if __name__ == "__main__":
    build_dataset()
