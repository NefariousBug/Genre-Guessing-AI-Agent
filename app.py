from flask import Flask, render_template, request, jsonify
import os
import yt_dlp
import joblib
from werkzeug.utils import secure_filename
from core.agent_runner import GenreAgentRunner

app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = 'uploads'
app.config['MAX_CONTENT_LENGTH'] = 50 * 1024 * 1024 

os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)

MODEL_PATH = "genre_model.pkl"
FEEDBACK_CSV = "feedback_data.csv"
GTZAN_CSV = "data/gtzan_features.csv"
VALID_GENRES = ["blues", "classical", "country", "disco", "hiphop", "jazz", "metal", "pop", "reggae", "rock"]

agent = GenreAgentRunner(MODEL_PATH, FEEDBACK_CSV, GTZAN_CSV)

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
        
        features_dict, predictions = agent.prediction_tick(file_path)
        
        if features_dict is None:
            if is_temp and os.path.exists(file_path):
                os.remove(file_path)
            return jsonify({'error': 'Could not process audio file'}), 400
        
        features_file = os.path.join(app.config['UPLOAD_FOLDER'], 'last_features.pkl')
        joblib.dump(features_dict, features_file)
        
        if is_temp and os.path.exists(file_path):
            os.remove(file_path)
        
        return jsonify({'predictions': predictions})
    
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
        
        agent.feedback_tick(features_dict, correct_genre)
        
        return jsonify({'message': 'Feedback saved successfully'})
    
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/retrain', methods=['POST'])
def retrain():
    try:
        feedback_count = agent.learning_tick()
        
        return jsonify({
            'message': 'Model retrained successfully', 
            'feedback_count': feedback_count
        })
    
    except Exception as e:
        return jsonify({'error': str(e)}), 500

if __name__ == '__main__':
    app.run(debug=True)