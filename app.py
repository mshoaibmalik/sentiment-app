from flask import Flask, render_template, request
import joblib
import os

app = Flask(__name__)

# Load model and vectorizer
model_path = os.path.join("sentiment_model", "model.pkl")
vectorizer_path = os.path.join("sentiment_model", "vectorizer.pkl")

model = joblib.load(model_path)
vectorizer = joblib.load(vectorizer_path)

# Define sentiment color mapping
def get_sentiment_color(sentiment):
    return {
        "Positive": "green",
        "Negative": "red",
        "Neutral": "orange"
    }.get(sentiment, "gray")

# Define model accuracy (you can dynamically compute this if needed)
MODEL_ACCURACY = "78.01%"

@app.route("/", methods=["GET", "POST"])
def index():
    sentiment = ""
    color = ""
    text = ""

    if request.method == "POST":
        text = request.form["text"]
        if text.strip():
            try:
                vect = vectorizer.transform([text])
                prediction = model.predict(vect)[0]
                sentiment = prediction
                color = get_sentiment_color(sentiment)
            except Exception as e:
                sentiment = "Prediction Error"
                color = "gray"
                print("Error:", e)
        else:
            sentiment = "Please enter text."
            color = "gray"

    return render_template("index.html", sentiment=sentiment, color=color, text=text, accuracy=MODEL_ACCURACY)

if __name__ == "__main__":
    app.run(debug=True)
