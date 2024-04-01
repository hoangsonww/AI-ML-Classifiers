import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report
from nltk.stem import WordNetLemmatizer
from nltk.corpus import stopwords
import nltk
import re

# Download NLTK data
nltk.download('wordnet')
nltk.download('stopwords')

# Sample data
data = {
    "text": ["I love this product", "This is a great movie", "I hate this movie", "This is a terrible product"],
    "sentiment": [1, 1, 0, 0]  # 1 for positive, 0 for negative
}

df = pd.DataFrame(data)

# Preprocessing function
def preprocess_text(text):
    # Lowercasing
    text = text.lower()
    # Remove non-alphanumeric characters
    text = re.sub(r'\W', ' ', text)
    # Remove extra spaces
    text = re.sub(r'\s+', ' ', text)
    # Initialize the WordNet Lemmatizer
    lemmatizer = WordNetLemmatizer()
    # Tokenization
    words = text.split()
    # Lemmatization and stop words removal
    words = [lemmatizer.lemmatize(word) for word in words if word not in stopwords.words('english')]
    return ' '.join(words)

# Apply preprocessing
df['processed_text'] = df['text'].apply(preprocess_text)

# Feature Extraction
vectorizer = TfidfVectorizer()
X = vectorizer.fit_transform(df['processed_text'])
y = df['sentiment']

# Split the dataset
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)
