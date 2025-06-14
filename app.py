from flask import Flask, render_template, request
import joblib
import os

app = Flask(__name__)

# Load model and vectorizer
model, vectorizer = joblib.load('model/sentiment_model.pkl')

# Load accuracy score
with open('model/accuracy.txt', 'r') as f:
    accuracy_score = f.read()

@app.route('/', methods=['GET', 'POST'])
def index():
    sentiment = ''
    warning = ''
    sentiment_color = ''

    if request.method == 'POST':
        text = request.form['text'].strip()
        if len(text) < 3:
            warning = "Please write some meaningful text to analyze."
        else:
            vector = vectorizer.transform([text])
            sentiment = model.predict(vector)[0]

            # Assign color based on sentiment
            if sentiment.lower() == 'positive':
                sentiment_color = 'success'
            elif sentiment.lower() == 'negative':
                sentiment_color = 'danger'
            else:
                sentiment_color = 'secondary'

    return render_template(
        'index.html',
        sentiment=sentiment,
        sentiment_color=sentiment_color,
        warning=warning,
        accuracy_score=accuracy_score
    )

if __name__ == '__main__':
    app.run(debug=True)
