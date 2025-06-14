import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import joblib
import os

# Load your dataset
data = pd.read_csv("data/cleaned_dataset.csv")  # Ensure the file contains 'text' and 'label' columns
data.dropna(inplace=True)

# Split data
X = data['Text']
y = data['Label']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Vectorizer and model
vectorizer = TfidfVectorizer(ngram_range=(1,2), max_features=5000)
X_train_vec = vectorizer.fit_transform(X_train)
X_test_vec = vectorizer.transform(X_test)

model = LogisticRegression()
model.fit(X_train_vec, y_train)

# Evaluate
y_pred = model.predict(X_test_vec)
accuracy = accuracy_score(y_test, y_pred)
print(f"Model Accuracy: {accuracy * 100:.2f}%")

# Save model and vectorizer
os.makedirs("sentiment_model", exist_ok=True)
joblib.dump(model, "sentiment_model/model.pkl")
joblib.dump(vectorizer, "sentiment_model/vectorizer.pkl")

