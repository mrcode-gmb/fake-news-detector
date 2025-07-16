📰 Fake News Detector
Project Overview
This project is a Machine Learning-based Fake News Detector built using Python, Scikit-Learn, and Flask.
It predicts whether a news article is Real or Fake with high confidence based on trained data.

Features
✅ Detects fake and real news articles

✅ Trained on a labeled dataset of real and fake news

✅ Web interface for user input using HTML & Tailwind CSS

✅ Instant prediction with clear result alerts

How It Works
The model was trained using a Logistic Regression classifier on a dataset containing real and fake news.

News articles are cleaned (removal of punctuations, numbers, etc.) and transformed using TF-IDF Vectorization.

The trained model predicts the class when a user inputs a news article.

The Flask app serves the model with a simple, user-friendly interface.

Technologies Used
🐍 Python (Scikit-Learn, Pandas)

🔥 Flask for the backend

🎨 Tailwind CSS for frontend styling

💾 Joblib for model saving

Disclaimer
While the model performs well on the dataset used for training and testing,
it is important to note that no fake news detector can guarantee 100% accuracy in real-world scenarios.
Always verify sensitive information from trusted sources.

Author
Developed by [Your Name] — 2025