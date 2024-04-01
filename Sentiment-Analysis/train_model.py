import os
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report
import joblib
import re
from nltk.stem import WordNetLemmatizer
from nltk.corpus import stopwords
import nltk
import time

print("Downloading necessary NLTK data...")
nltk.download('wordnet')
nltk.download('stopwords')

print("Loading the dataset...")
dataset_path = 'Sentiment-Analysis/training.1600000.processed.noemoticon.csv'
columns = ['sentiment', 'id', 'date', 'query', 'user', 'text']
df = pd.read_csv(dataset_path, encoding='latin-1', names=columns, usecols=['sentiment', 'text'], low_memory=False, dtype={'sentiment': str})

print("Preprocessing the sentiment labels...")
df['sentiment'] = df['sentiment'].apply(lambda x: 1 if x.strip() == '4' else 0)

def preprocess_text(text):
    text = text.lower()
    text = re.sub(r'\W', ' ', text)
    text = re.sub(r'\s+', ' ', text).strip()
    lemmatizer = WordNetLemmatizer()
    words = text.split()
    filtered_words = [lemmatizer.lemmatize(word) for word in words if word not in stopwords.words('english')]
    return ' '.join(filtered_words)

print("Applying text preprocessing...")
total = len(df)
batch_size = total // 10  # Divide the data into 10 batches
processed_texts = []

for i in range(0, total, batch_size):
    batch = df['text'][i:i+batch_size].apply(preprocess_text)
    processed_texts.extend(batch)
    print(f"Preprocessed {(i+batch_size) / total * 100:.0f}% of the data...")

df['processed_text'] = processed_texts

print("Extracting features with TF-IDF...")
vectorizer = TfidfVectorizer(max_features=1000)  # Limit the number of features to speed up training
X = vectorizer.fit_transform(df['processed_text'])
y = df['sentiment']

print("Splitting the dataset into training and testing sets...")
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

print("Initializing the Logistic Regression model...")
model = LogisticRegression(random_state=42, max_iter=100, solver='saga', n_jobs=-1)  # Use all CPU cores

print("Starting model training...")
start_time = time.time()
model.fit(X_train, y_train)
end_time = time.time()

print(f"Training completed in {end_time - start_time:.2f} seconds.")

print("Evaluating the model...")
predictions = model.predict(X_test)
print(classification_report(y_test, predictions))

# Define the directory to save the model and vectorizer
save_dir = 'Sentiment-Analysis'
if not os.path.exists(save_dir):
    print(f"Creating directory {save_dir} for model and vectorizer...")
    os.makedirs(save_dir)

print("Saving the model and vectorizer...")
model_file_path = os.path.join(save_dir, 'sentiment_model.pkl')
vectorizer_file_path = os.path.join(save_dir, 'vectorizer.pkl')
joblib.dump(model, model_file_path)
joblib.dump(vectorizer, vectorizer_file_path)

print(f"Model saved to {model_file_path}")
print(f"Vectorizer saved to {vectorizer_file_path}")
