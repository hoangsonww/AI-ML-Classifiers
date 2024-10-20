import os
import pandas as pd
import nltk
import time
import joblib
import re

from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, accuracy_score
from nltk.stem import WordNetLemmatizer
from nltk.corpus import stopwords
from tqdm import tqdm


nltk.download('wordnet')
nltk.download('stopwords')


# Choose the dataset size
dataset_choice = input("Choose the dataset size (small/large): ").lower().strip()
if dataset_choice == 'small':
    dataset_path = 'Sentiment-Analysis/small_dataset.csv'
elif dataset_choice == 'large':
    dataset_path = 'Sentiment-Analysis/training.1600000.processed.noemoticon.csv'
else:
    print("Invalid choice. Please choose 'small' or 'large'.")
    exit(1)


# Load the dataset
print(f"Loading the {dataset_choice} dataset...")
df = pd.read_csv(dataset_path, encoding='latin-1', header=None,
                 names=['sentiment', 'id', 'date', 'query', 'user', 'text'], low_memory=False)
df['sentiment'] = df['sentiment'].apply(lambda x: 1 if x == '4' or x == 4 else 0)


# Preprocess text data
print("Applying text preprocessing...")
lemmatizer = WordNetLemmatizer()
cached_stopwords = stopwords.words('english')


# Define the text preprocessing function
def preprocess_text(text):
    text = text.lower()
    text = re.sub(r'\W', ' ', text)
    text = re.sub(r'\s+', ' ', text).strip()
    words = text.split()
    filtered_words = [lemmatizer.lemmatize(word) for word in words if word not in cached_stopwords]
    return ' '.join(filtered_words)


total = len(df)
processed_texts = []
progress_step = total // 20

print("Starting text preprocessing...")
for i, text in enumerate(df['text']):
    processed_texts.append(preprocess_text(text))
    if (i + 1) % progress_step == 0:
        progress = (i + 1) / total * 100
        print(f"Preprocessing progress: {progress:.2f}%")


df['processed_text'] = processed_texts
print("Text preprocessing completed.")


# Extract features using TF-IDF
print("Extracting features with TF-IDF...")
vectorizer = TfidfVectorizer(max_features=10000, ngram_range=(1, 2))
processed_texts_tqdm = tqdm(df['processed_text'], total=len(df['processed_text']), desc="TF-IDF", unit='texts',
                            mininterval=1.0)
X = vectorizer.fit_transform(processed_texts_tqdm)
y = df['sentiment'].astype(int)
print("Feature extraction completed.")


# Split the dataset into training and testing sets
print("Splitting the dataset into training and testing sets...")
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
print("Dataset splitting completed.")


# Initialize and train the Logistic Regression model
print("Initializing and training the Logistic Regression model...")
model = LogisticRegression(random_state=42, max_iter=1000, solver='lbfgs', n_jobs=-1)

start_time = time.time()
model.fit(X_train, y_train)
end_time = time.time()
print(f"Training completed in {end_time - start_time:.2f} seconds.")

predictions = model.predict(X_test)
print(classification_report(y_test, predictions))


# Model evaluation
expected_accuracy = model.score(X_test, y_test)
print(f"Expected accuracy of the model: {expected_accuracy * 100:.2f}%")

prediction_probabilities = model.predict_proba(X_test)


# Calculate the expected confidence level
expected_confidence_level = prediction_probabilities.max(axis=1).mean()
print(f"Expected confidence level of the model: {expected_confidence_level * 100:.2f}%")


# Save the model and vectorizer
save_dir = 'Sentiment-Analysis'
os.makedirs(save_dir, exist_ok=True)

model_file_path = os.path.join(save_dir, 'sentiment_model.pkl')
vectorizer_file_path = os.path.join(save_dir, 'vectorizer.pkl')
joblib.dump(model, model_file_path)
joblib.dump(vectorizer, vectorizer_file_path)

print(f"Model saved to {model_file_path}")
print(f"Vectorizer saved to {vectorizer_file_path}")
print("Training completed successfully.")
