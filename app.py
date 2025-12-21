from flask import Flask, render_template, request, jsonify
import os
import librosa
import numpy as np
import joblib
import pandas as pd
import yt_dlp
from werkzeug.utils import secure_filename

app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = 'uploads'
app.config['MAX_CONTENT_LENGTH'] = 50 * 1024 * 1024  # 50mb max

os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)

MODEL_PATH = "genre_model.pkl"
FEEDBACK_CSV = "feedback_data.csv"
VALID_GENRES = ["blues", "classical", "country", "disco", "hiphop", "jazz", "metal", "pop", "reggae", "rock"]

model = joblib.load(MODEL_PATH)

def download_audio_from_url(url):
    temp_file = os.path.join(app.config['UPLOAD_FOLDER'], 'temp_download')
    
    ydl_opts = {
        'format': 'bestaudio/best',
        'outtmpl': f'{temp_file}.%(ext)s',
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
    
    return f"{temp_file}.wav"

def extract_features_dict(file_path):
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

@app.route('/')
def index():
    return render_template('index.html', genres=VALID_GENRES)

@app.route('/predict', methods=['POST'])
def predict():
    try:
        file_path = None
        is_temp = False
        
        if 'youtube_url' in request.form and request.form['youtube_url'].strip():
            url = request.form['youtube_url'].strip()
            try:
                file_path = download_audio_from_url(url)
                is_temp = True
            except Exception as e:
                return jsonify({'error': f'Failed to download audio: {str(e)}'}), 400
        
        elif 'audio_file' in request.files:
            file = request.files['audio_file']
            if file.filename == '':
                return jsonify({'error': 'No file selected'}), 400
            
            filename = secure_filename(file.filename)
            file_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
            file.save(file_path)
            is_temp = True
        
        else:
            return jsonify({'error': 'Please provide either a YouTube URL or upload a file'}), 400
        
        features_dict = extract_features_dict(file_path)
        if features_dict is None:
            if is_temp and os.path.exists(file_path):
                os.remove(file_path)
            return jsonify({'error': 'Could not process audio file'}), 400
        
        # prediction
        expected_features = model.named_steps['scaler'].feature_names_in_
        X_new = pd.DataFrame([features_dict])
        X_new = X_new[expected_features]
        
        probabilities = model.predict_proba(X_new)[0]
        all_genres = model.classes_
        
        genre_prob_pairs = sorted(zip(all_genres, probabilities), key=lambda x: x[1], reverse=True)
        top_3 = [{'genre': genre, 'probability': float(prob * 100)} for genre, prob in genre_prob_pairs[:3]]
        
        # Store features in temp file
        features_file = os.path.join(app.config['UPLOAD_FOLDER'], 'last_features.pkl')
        joblib.dump(features_dict, features_file)
        
        if is_temp and os.path.exists(file_path):
            os.remove(file_path)
        
        return jsonify({'predictions': top_3})
    
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/feedback', methods=['POST'])
def feedback():
    try:
        data = request.json
        correct_genre = data.get('genre')
        
        if correct_genre not in VALID_GENRES:
            return jsonify({'error': 'Invalid genre'}), 400
        
        features_file = os.path.join(app.config['UPLOAD_FOLDER'], 'last_features.pkl')
        if not os.path.exists(features_file):
            return jsonify({'error': 'No prediction to give feedback on'}), 400
        
        features_dict = joblib.load(features_file)
        save_feedback(features_dict, correct_genre)
        
        return jsonify({'message': 'Feedback saved successfully!'})
    
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/retrain', methods=['POST'])
def retrain():
    global model
    try:
        from sklearn.model_selection import train_test_split
        from sklearn.preprocessing import StandardScaler
        from sklearn.pipeline import Pipeline
        from sklearn.svm import SVC
        
        GTZAN_CSV = "data/gtzan_features.csv"
        FEEDBACK_WEIGHT = 3.0
        
        # GTZAN
        if not os.path.exists(GTZAN_CSV):
            return jsonify({'error': 'GTZAN dataset not found'}), 400
        
        df_gtzan = pd.read_csv(GTZAN_CSV)
        
        # feedback
        if os.path.exists(FEEDBACK_CSV):
            df_feedback = pd.read_csv(FEEDBACK_CSV)
        else:
            return jsonify({'error': 'No feedback data to retrain on'}), 400
        
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
        
        joblib.dump(new_model, MODEL_PATH)
        
        model = joblib.load(MODEL_PATH)
        
        return jsonify({'message': 'Model retrained successfully', 'feedback_count': len(df_feedback)})
    
    except Exception as e:
        return jsonify({'error': str(e)}), 500

if __name__ == '__main__':
    app.run(debug=True)