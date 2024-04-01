# use_model.py
import joblib
from nltk.stem import WordNetLemmatizer
from nltk.corpus import stopwords
import nltk
nltk.download('wordnet')
nltk.download('stopwords')
import re

# Ensure the necessary NLTK data is downloaded
nltk.download('wordnet')
nltk.download('stopwords')

# Load the model and vectorizer
model = joblib.load('sentiment_model.pkl')
vectorizer = joblib.load('vectorizer.pkl')

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
    prediction = model.predict(vectorized_text)
    return "Positive" if prediction[0] == 1 else "Negative"

# Example usage
if __name__ == "__main__":
    text_input = input("Enter a sentence to analyze its sentiment: ")
    sentiment = predict_sentiment(text_input)
    print(f"The sentiment of the input text is: {sentiment}")
