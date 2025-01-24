from flask import Flask, render_template, request
import joblib
import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer

# Initialize Flask app
app = Flask(__name__)

# Load model and vectorizer
model_data = joblib.load('model_svm_with_vectorizer.pkl')
model_svm = model_data['model']
tfidf_vectorizer = model_data['vectorizer']

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    if request.method == 'POST':
        # Get user input from form
        user_input = request.form['user_input']
        
        # Preprocess input (same as preprocessing in your model)
        input_tfidf = tfidf_vectorizer.transform([user_input])

        # Make prediction using SVM model
        prediction = model_svm.predict(input_tfidf)[0]
        
        # Map prediction to sentiment label (assuming three classes: 0, 1, 2)
        sentiment_map = {0: 'negatif', 1: 'netral', 2: 'positif'}
        sentiment = sentiment_map.get(prediction, f": {prediction}")
        
        # Pass the user input and sentiment result to template
        return render_template('index.html', prediction_text=f'Sentiment {sentiment}', user_input=user_input)

if __name__ == "__main__":
    app.run(debug=True)