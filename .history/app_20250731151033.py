from flask import Flask, render_template, request
import joblib
import pandas as pd
import random
import re, string

app = Flask(__name__)

# Load model and vectorizer
model = joblib.load("model/model.pkl")
vectorizer = joblib.load("model/vectorizer.pkl")

# Load sample dataset for explanations
df = pd.read_csv("model/news_dataset.csv")  # dataset must have 'text' and 'label' columns

def clean_text(text):
    text = text.lower()
    text = text.translate(str.maketrans('', '', string.punctuation))
    text = re.sub(r'\d+', '', text)
    text = re.sub(r'\s+', ' ', text).strip()
    return text

@app.route('/')
def home():
    return render_template("index.html")

@app.route('/predict', methods=["POST"])
def predict():
    if request.method == "POST":
        news = request.form['news']
        cleaned_news = clean_text(news)
        vectorized_news = vectorizer.transform([cleaned_news])
        prediction = model.predict(vectorized_news)[0]

        label = "Real News" if prediction == 1 else "Fake News"

        # Suggest reason based on similar data
        similar_examples = df[df['label'] == prediction].sample(n=1)
        example_text = similar_examples['text'].values[0][:300]  # trim for display

        explanation = f"This news is predicted as *{label}*. Here's a similar example from the dataset:\n\n\"{example_text}...\""

        return render_template("index.html", prediction=label, explanation=explanation)

if __name__ == "__main__":
    app.run(debug=True)
