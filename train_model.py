import pandas as pd
import joblib

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score

# Load dataset
df = pd.read_csv("data/gtzan_features.csv")

# Separate features and labels
X = df.drop("genre", axis=1)
y = df["genre"]

# Train / test split
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

# Pipeline: scaling + model
model = Pipeline([
    ("scaler", StandardScaler()),
    ("classifier", SVC(kernel="rbf", probability=True))
])

# Train
model.fit(X_train, y_train)

# Evaluate
y_pred = model.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)

print(f"Model accuracy: {accuracy:.2f}")

# Save model
joblib.dump(model, "genre_model.pkl")
print("Model saved as genre_model.pkl")
