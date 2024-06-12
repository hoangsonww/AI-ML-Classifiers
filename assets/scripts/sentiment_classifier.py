import joblib
import nltk
import re

from nltk.stem import WordNetLemmatizer
from nltk.corpus import stopwords

print("Running sentiment analysis...")

try:
    nltk.data.find('corpora/wordnet')
    nltk.data.find('corpora/stopwords')
except LookupError:
    nltk.download('wordnet')
    nltk.download('stopwords')

model = joblib.load('sentiment_model.pkl')
vectorizer = joblib.load('vectorizer.pkl')

print("Model loaded successfully.")


def preprocess_text(text):
    text = text.lower()
    text = re.sub(r'\W', ' ', text)
    text = re.sub(r'\s+', ' ', text).strip()
    lemmatizer = WordNetLemmatizer()
    words = text.split()
    filtered_words = [lemmatizer.lemmatize(word) for word in words if word not in stopwords.words('english')]
    return ' '.join(filtered_words)


def predict_sentiment(text):
    processed_text = preprocess_text(text)
    vectorized_text = vectorizer.transform([processed_text])
    prediction_proba = model.predict_proba(vectorized_text)
    prediction = prediction_proba.argmax(axis=1)
    confidence = prediction_proba.max(axis=1)[0]
    sentiment = "Positive" if prediction[0] == 1 else "Negative"
    return sentiment, confidence * 100  # Convert to percentage


if __name__ == "__main__":
    text_input = input("Enter a sentence to analyze its sentiment: ")
    sentiment, confidence = predict_sentiment(text_input)
    print(f"The sentiment/emotion of the input text is: {sentiment} with {confidence:.2f}% confidence")
