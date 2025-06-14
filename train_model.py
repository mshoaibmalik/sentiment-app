import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import accuracy_score
import joblib
import os

# Load dataset
df = pd.read_csv('data/cleaned_dataset.csv')
df.dropna(subset=['Text', 'Label'], inplace=True)

# Prepare data
X = df['Text']
y = df['Label']

vectorizer = CountVectorizer()
X_vec = vectorizer.fit_transform(X)

# Train/test split
X_train, X_test, y_train, y_test = train_test_split(X_vec, y, test_size=0.2)

# Train model
model = MultinomialNB()
model.fit(X_train, y_train)

# Evaluate accuracy
y_pred = model.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)

# Save model and vectorizer
joblib.dump((model, vectorizer), 'model/sentiment_model.pkl')

# Save accuracy score to file
with open('model/accuracy.txt', 'w') as f:
    f.write(str(round(accuracy * 100, 2)))

print(f"Model trained with accuracy: {accuracy * 100:.2f}%")
