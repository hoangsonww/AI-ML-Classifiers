import pandas as pd
import random


def generate_small_dataset(filename, n=1000):
    sentiments = ['positive', 'negative']
    words_positive = ['good', 'fantastic', 'amazing', 'love', 'great', 'excellent', 'happy']
    words_negative = ['bad', 'terrible', 'awful', 'hate', 'poor', 'disappointing', 'sad']

    data = []
    for _ in range(n):
        sentiment = random.choice(sentiments)
        text = ' '.join(random.sample(words_positive if sentiment == 'positive' else words_negative, 3))
        numeric_sentiment = '4' if sentiment == 'positive' else '0'

        # Create a dummy row with the expected number of columns
        row = [numeric_sentiment, 'id', 'date', 'query', 'user', text]
        data.append(row)

    # Define the columns as expected by the training script
    columns = ['sentiment', 'id', 'date', 'query', 'user', 'text']
    df = pd.DataFrame(data, columns=columns)

    # Save the dataset with no index and ensure UTF-8 encoding
    df.to_csv(filename, index=False, encoding='utf-8')


generate_small_dataset('Sentiment-Analysis/small_dataset.csv')
