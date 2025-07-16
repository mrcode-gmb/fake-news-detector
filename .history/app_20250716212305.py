from flask import Flask, render_template, request
import joblib

app = Flask(__name__)

# Load the trained model and vectorizer
model = joblib.load("model/model.pkl")
vectorizer = joblib.load("model/vectorizer.pkl")

@app.route('/')
def home():
    return render_template("index.html")

def clean_text(text):
    import re, string
    text = text.lower()
    text = text.translate(str.maketrans('', '', string.punctuation))
    text = re.sub(r'\d+', '', text)
    text = re.sub(r'\s+', ' ', text).strip()
    return text
@app.route('/predict', methods=["POST"])


def predict():
    if request.method == "POST":
        news = request.form['news']
        cleaned_news = clean_text(news)
        vectorized_news = vectorizer.transform([cleaned_news])

        prediction = model.predict(vectorized_news)[0]
        label = "Real News" if prediction == 1 else "Fake News"
        return render_template("index.html", prediction=label)

if __name__ == "__main__":
    app.run(debug=True)
