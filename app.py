from flask import Flask, render_template, request, jsonify
import joblib
from flask_cors import CORS

app = Flask(__name__)
CORS(app)  # Enable CORS for all routes

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
@app.route('/api/predict', methods=["POST"])
def api_predict():
    if not request.is_json:
        return jsonify({
            'success': False,
            'error': 'Content-Type must be application/json'
        }), 400
        
    data = request.get_json()
    if 'news' not in data or not data['news'].strip():
        return jsonify({
            'success': False,
            'error': 'News text is required'
        }), 400
        
    try:
        cleaned_news = clean_text(data['news'])
        vectorized_news = vectorizer.transform([cleaned_news])
        prediction = model.predict(vectorized_news)[0]
        
        return jsonify({
            'success': True,
            'prediction': 'real' if prediction == 1 else 'fake',
            'confidence': float(max(model.predict_proba(vectorized_news)[0])),
            'text': cleaned_news[:500] + '...' if len(cleaned_news) > 500 else cleaned_news
        })
    except Exception as e:
        return jsonify({
            'success': False,
            'error': str(e)
        }), 500

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