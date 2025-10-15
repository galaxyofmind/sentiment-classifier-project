from flask import Flask, render_template, request
import joblib
import logging
from sklearn.feature_extraction.text import TfidfVectorizer
import pandas as pd
from pipeline_v2.preprocessNLP import Preprocessor
import underthesea
from pipeline_v2.vectorization import Vectorizer_TFIDF

app = Flask(__name__)

# Set up logging
logging.basicConfig(
    level=logging.DEBUG,
    format='%(asctime)s %(levelname)s %(message)s',
    handlers=[logging.StreamHandler()]
)


# Init preprocess
preprocess = Preprocessor(stopwords_file="dataset/vietnamese-stopwords.txt")

# Load models
model = joblib.load("model/best.pkl")
vectorizer = joblib.load("model/vectorizer.pkl")

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    data = request.get_json()
    text = data.get('text')
    model_choice = data.get('model')
    logging.debug(f"Received text: {text}")
    logging.debug(f"Model selected: {model_choice}")

    # Preprocessing for input
    text = preprocess.lower_strip(text)
    text = preprocess.normalize(text)
    tokens = preprocess.tokenize(text)
    tokens = preprocess.lemmatizer(tokens)
    tokens = " ".join(tokens).strip()
    text = underthesea.word_tokenize(tokens, format="text")

    # vectorizer = model.vectorizer
    X_vec = vectorizer.transform([text]) 

    prediction = model.predict(X_vec)

    if prediction[0] == 1:
        prediction = "Tích cực"
    else:
        prediction = "Tiêu cực"

    return {"prediction": prediction}



if __name__ == '__main__':
    app.run(debug=True)