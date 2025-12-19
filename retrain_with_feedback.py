import pandas as pd
import joblib
import os
import numpy as np

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score

GTZAN_CSV = "data/gtzan_features.csv"
FEEDBACK_CSV = "feedback_data.csv"
MODEL_PATH = "genre_model.pkl"
FEEDBACK_WEIGHT = 5.0  # Each feedback sample counts as 5 GTZAN samples

def retrain_model():
    print("=== Retraining Model with Feedback ===\n")
    
    # 1. Load original GTZAN data
    if not os.path.exists(GTZAN_CSV):
        print(f"ERROR: {GTZAN_CSV} not found!")
        return
    
    df_gtzan = pd.read_csv(GTZAN_CSV)
    print(f"✓ Loaded {len(df_gtzan)} samples from GTZAN dataset")
    
    # 2. Load feedback data (if exists)
    if os.path.exists(FEEDBACK_CSV):
        df_feedback = pd.read_csv(FEEDBACK_CSV)
        print(f"✓ Loaded {len(df_feedback)} feedback samples")
    else:
        print("⚠ No feedback data found. Training on GTZAN only.")
        df_feedback = pd.DataFrame()
    
    # 3. Combine datasets
    df_combined = pd.concat([df_gtzan, df_feedback], ignore_index=True)
    print(f"✓ Combined dataset: {len(df_combined)} total samples\n")
    
    # 4. Create sample weights
    # GTZAN samples get weight 1.0, feedback samples get FEEDBACK_WEIGHT
    weights = np.ones(len(df_combined))
    if len(df_feedback) > 0:
        # The last len(df_feedback) samples are from feedback
        weights[-len(df_feedback):] = FEEDBACK_WEIGHT
        print(f"✓ Feedback samples weighted {FEEDBACK_WEIGHT}x more than GTZAN samples\n")
    
    # 5. Separate features and labels
    X = df_combined.drop("genre", axis=1)
    y = df_combined["genre"]
    
    # 6. Train/test split (stratified)
    X_train, X_test, y_train, y_test, weights_train, weights_test = train_test_split(
        X, y, weights, test_size=0.2, random_state=42, stratify=y
    )
    
    # 7. Build and train model with sample weights
    print("Training model...")
    model = Pipeline([
        ("scaler", StandardScaler()),
        ("classifier", SVC(kernel="rbf", probability=True))
    ])
    
    model.fit(X_train, y_train, classifier__sample_weight=weights_train)
    
    # 8. Evaluate
    y_pred = model.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    
    print(f"✓ Model accuracy: {accuracy:.2%}\n")
    
    # 9. Save updated model
    joblib.dump(model, MODEL_PATH)
    print(f"✓ Model saved as {MODEL_PATH}")
    print("\n=== Retraining Complete! ===")

if __name__ == "__main__":
    retrain_model()