from flask import Flask, render_template, request
import joblib
import re
import os

app = Flask(__name__)

# Load model
MODEL_PATH = os.path.join(os.path.dirname(__file__), '..', 'models')
model = joblib.load(os.path.join(MODEL_PATH, 'model.pkl'))
tfidf = joblib.load(os.path.join(MODEL_PATH, 'tfidf.pkl'))

def clean_text(text):
    text = str(text).lower().strip()
    text = re.sub(r'http\S+|www\S+|https\S+', '', text)
    text = re.sub(r'[^a-z ]', ' ', text)
    text = re.sub(r'\s+', ' ', text)
    return text  

@app.route('/', methods=['GET', 'POST'])
def index():
    result = None
    if request.method == 'POST':
        review = request.form['review']
        cleaned = clean_text(review)
        
        vec = tfidf.transform([cleaned])
        pred = model.predict(vec)[0]
        prob = model.predict_proba(vec)[0].max()
        
        result = {
            'sentiment': 'Positive' if pred == 1 else 'Negative',
            'confidence': f"{prob:.0%}",
            'cleaned': cleaned[:100] + '...'
        }
    return render_template('index.html', result=result)

if __name__ == '__main__':
    app.run(debug=True,host='0.0.0.0', port=5000)
