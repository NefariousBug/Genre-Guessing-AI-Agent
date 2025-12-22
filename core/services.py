import os
import librosa
import numpy as np
import pandas as pd
import joblib
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.svm import SVC

FEEDBACK_WEIGHT = 3.0

class AudioFeatureService:
    
    @staticmethod
    def extract_features(file_path):
        if not os.path.exists(file_path):
            return None

        try:
            total_duration = librosa.get_duration(path=file_path)

            if total_duration > 30:
                start_time = (total_duration / 2) - 15
            else:
                start_time = 0

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

        # 3. Spectral features
        features["spectral_centroid"] = np.mean(librosa.feature.spectral_centroid(y=y, sr=sr))
        features["spectral_bandwidth"] = np.mean(librosa.feature.spectral_bandwidth(y=y, sr=sr))
        features["spectral_rolloff"] = np.mean(librosa.feature.spectral_rolloff(y=y, sr=sr))
        features["zero_crossing_rate"] = np.mean(librosa.feature.zero_crossing_rate(y))

        # 4. Tempo
        tempo, _ = librosa.beat.beat_track(y=y, sr=sr)
        features["tempo"] = float(tempo[0])

        return features


class PredictionService:
    
    def __init__(self, model_path):
        self.model_path = model_path
        self.model = joblib.load(model_path)
    
    def predict(self, features_dict):
        expected_features = self.model.named_steps['scaler'].feature_names_in_
        X_new = pd.DataFrame([features_dict])
        X_new = X_new[expected_features]
        
        probabilities = self.model.predict_proba(X_new)[0]
        all_genres = self.model.classes_
        
        genre_prob_pairs = sorted(
            zip(all_genres, probabilities), 
            key=lambda x: x[1], 
            reverse=True
        )
        
        return [
            {'genre': genre, 'probability': float(prob * 100)} 
            for genre, prob in genre_prob_pairs[:3]
        ]
    
    def reload_model(self):
        self.model = joblib.load(self.model_path)


class FeedbackService:
    
    def __init__(self, feedback_csv):
        self.feedback_csv = feedback_csv
    
    def save_feedback(self, features_dict, correct_genre):
        features_dict["genre"] = correct_genre
        feedback_row = pd.DataFrame([features_dict])
        
        if os.path.exists(self.feedback_csv):
            feedback_row.to_csv(self.feedback_csv, mode='a', header=False, index=False)
        else:
            feedback_row.to_csv(self.feedback_csv, mode='w', header=True, index=False)


class RetrainingService:

    def __init__(self, gtzan_csv, feedback_csv, model_path):
        self.gtzan_csv = gtzan_csv
        self.feedback_csv = feedback_csv
        self.model_path = model_path
    
    def should_retrain(self):
        return os.path.exists(self.feedback_csv)
    
    def retrain(self):
        if not os.path.exists(self.gtzan_csv):
            raise FileNotFoundError("GTZAN dataset not found")
        
        df_gtzan = pd.read_csv(self.gtzan_csv)
        
        if not os.path.exists(self.feedback_csv):
            raise FileNotFoundError("No feedback data to retrain on")
        
        df_feedback = pd.read_csv(self.feedback_csv)
        
        df_combined = pd.concat([df_gtzan, df_feedback], ignore_index=True)
        
        weights = np.ones(len(df_combined))
        weights[-len(df_feedback):] = FEEDBACK_WEIGHT
        
        X = df_combined.drop("genre", axis=1)
        y = df_combined["genre"]
        
        X_train, X_test, y_train, y_test, weights_train, weights_test = train_test_split(
            X, y, weights, test_size=0.2, random_state=42, stratify=y
        )
        
        new_model = Pipeline([
            ("scaler", StandardScaler()),
            ("classifier", SVC(kernel="rbf", probability=True))
        ])
        
        new_model.fit(X_train, y_train, classifier__sample_weight=weights_train)
        
        joblib.dump(new_model, self.model_path)
        
        return len(df_feedback)