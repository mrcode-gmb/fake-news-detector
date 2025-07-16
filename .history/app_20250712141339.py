from flask import Flask, render_template, request
import joblib

app = Flask(__name__)

# Load the trained model and vectorizer
model = joblib.load("model/model.pkl")
vectorizer = joblib.load("model/vectorizer.pkl")

@app.route('/')
def home():
    return render_template("index.html")

@app.route('/predict', methods=["POST"])
def predict():
    if request.method == "POST":
        news = request.form['news']
        data = vectorizer.transform([news])
        prediction = model.predict(data)[0]

        label = "Real News" if prediction == 0 else "Fake News"
        return render_template("index.html", prediction=label)

if __name__ == "__main__":
    app.run(debug=True)
