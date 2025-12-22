from core.services import (
    AudioFeatureService, 
    PredictionService, 
    FeedbackService, 
    RetrainingService
)

class GenreAgentRunner:
 
    # Sense → Think → Act → Learn
  
    def __init__(self, model_path, feedback_csv, gtzan_csv):
        self.feature_service = AudioFeatureService()
        self.prediction_service = PredictionService(model_path)
        self.feedback_service = FeedbackService(feedback_csv)
        self.retraining_service = RetrainingService(gtzan_csv, feedback_csv, model_path)
    
    # low-level metode

    def sense(self, audio_file_path):
        """
        SENSE: Extract feature-a iz audio file-a (percepcija)
        Vraca feature dictionary ili None u slucaju nesupjeha
        """
        return self.feature_service.extract_features(audio_file_path)
    
    def think(self, features_dict):
        """
        THINK: Pravi predikciju na osnovu feature-a (policy)
        Vraca top 3 listu sa postocima
        """
        return self.prediction_service.predict(features_dict)
    
    def act_feedback(self, features_dict, correct_genre):
        """
        ACT: Sprema user feedback (akcija u okruzenju)
        """
        self.feedback_service.save_feedback(features_dict, correct_genre)
    
    def learn(self):
        """
        LEARN: Retreniranje modela sa prikupljenim feedback-om
        Vraca informaciju o tome koliko je feedback primjeraka koristeno
        """
        if not self.retraining_service.should_retrain():
            raise ValueError("No feedback data available for retraining")
        
        feedback_count = self.retraining_service.retrain()
        
        self.prediction_service.reload_model()
        
        return feedback_count
    
   #high-level metode

    def prediction_tick(self, audio_file_path):
        """
        Prediction ciklus, iz sense u think
        """
        features = self.sense(audio_file_path)
        if features is None:
            return None, None
        
        predictions = self.think(features)
        
        return features, predictions
    
    def feedback_tick(self, features_dict, correct_genre):
        """
        Feedback, act
        """
        self.act_feedback(features_dict, correct_genre)
    
    def learning_tick(self):
        """
        Learn ciklus, koristenje feedback-a za retreniranje
        """
        return self.learn()