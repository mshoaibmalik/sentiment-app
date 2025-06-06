from flask import Flask, render_template, request
import joblib

app = Flask(__name__)
model = joblib.load('model/sentiment_model.pkl')

@app.route('/', methods=['GET', 'POST'])
def index():
    sentiment = ""
    warning = ""
    if request.method == 'POST':
        text = request.form['text'].strip()
        if not text or len(text.split()) < 2:
            warning = "Please write some meaningful text to analyze."
        else:
            sentiment = model.predict([text])[0]
    return render_template('index.html', sentiment=sentiment, warning=warning)

if __name__ == '__main__':
    print("Starting Flask app...")
    app.run(host='0.0.0.0', port=5000)
