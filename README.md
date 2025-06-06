# Sentiment Analysis Web App

This is a simple web application built with **Flask** that uses a pre-trained **machine learning model** to analyze the **sentiment** of input text. It classifies the input as Positive, Negative, or Neutral.

---

## 🔍 Features

- Text input for sentiment analysis
- Feedback if the input is too short or empty
- Displays sentiment result in real time
- Built with Python, Flask, and scikit-learn
- Deployable on platforms like **Vercel**

---

## 🧠 How It Works

- The app loads a trained sentiment analysis model from `model/sentiment_model.pkl`
- User inputs a sentence or paragraph
- The model predicts the sentiment class
- The result is rendered on the same page using a Jinja2 template

