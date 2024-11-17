'''
Project A: Predicting Sentiment from Tweets

The problem is to determine whether a given Tweet has a positive or negative sentiment.
This dataset contains 10,000 highly polar Tweets (50% positive and 50% negative).
'''

import pandas as pd
import matplotlib.pyplot as plt
import nltk
from nltk.corpus import stopwords
import re
from nltk.stem import PorterStemmer, WordNetLemmatizer

# Download NLTK resources
nltk.download('punkt_tab')      # punkt tokenizer
nltk.download('stopwords')      # NLTK stopwords
nltk.download('wordnet')        # WordNet lemmatizer
nltk.download('omw-1.4')        # Lexical data for lemmatization
stop_words = set(stopwords.words('english'))

# Extra stopwords specific to Twitter text
extra_stopwords = {'u', 'im', 'rt', 'us', 'like', 'just', 'amp', 'dont', 'get', 'got', 'go', 'one', 'would', 'could'}
stop_words.update(extra_stopwords)

# Load and Merge the JSON Files
json_files = ['twitter_samples/negative_tweets.json', 'twitter_samples/positive_tweets.json']

# Merge into a single DataFrame
dataframes = [pd.read_json(file, lines=True) for file in json_files]
data = pd.concat(dataframes, ignore_index=True)

# Calculate tweet lengths for distribution analysis
data['text_length'] = data['text'].apply(lambda x: len(x) if pd.notnull(x) else 0)

# Plot distribution of tweet lengths
plt.figure(figsize=(10, 5))
plt.hist(data['text_length'], bins=30, color='skyblue', edgecolor='black')
plt.title('Distribution of Tweet Lengths')
plt.xlabel('Tweet Length')
plt.ylabel('Frequency')
plt.show()

# Function for Cleaning Text
def clean_text(text):
    text = re.sub(r'http\S+|www\S+|https\S+', '', text, flags=re.MULTILINE)
    text = re.sub(r'@\w+|#', '', text)  # Remove mentions and hashtags
    text = re.sub(r'[^a-zA-Z\s]', '', text)  # Keep only letters and spaces
    text = text.lower()  # Convert to lowercase
    return text

# Clean the tweets
data['clean_text'] = data['text'].apply(lambda x: clean_text(x) if pd.notnull(x) else '')

### DATA PREPROCESSING ###

# Tokenization
data['tokens'] = data['clean_text'].apply(nltk.word_tokenize)

# Remove stopwords from tokens
data['tokens'] = data['tokens'].apply(lambda x: [word for word in x if word not in stop_words])

# Initialize Stemmer and Lemmatizer
stemmer = PorterStemmer()
lemmatizer = WordNetLemmatizer()

# Stemming
data['stemmed_tokens'] = data['tokens'].apply(lambda x: [stemmer.stem(word) for word in x])

# Lemmatization
data['lemmatized_tokens'] = data['tokens'].apply(lambda x: [lemmatizer.lemmatize(word) for word in x])

# Print sample of cleaned, stemmed, and lemmatized text
print(data[['clean_text', 'tokens', 'stemmed_tokens', 'lemmatized_tokens']].head())

# Extract token frequency distribution
all_tokens = [token for sublist in data['lemmatized_tokens'] for token in sublist]
token_freq = nltk.FreqDist(all_tokens)

# Extract the most common tokens and their frequencies
most_common_tokens = token_freq.most_common(10)
tokens, counts = zip(*most_common_tokens)

# Plot the most common tokens
plt.figure(figsize=(10, 6))
plt.bar(tokens, counts, color='skyblue', edgecolor='black')
plt.title('Top 10 Most Common Tokens After Cleaning')
plt.xlabel('Tokens')
plt.ylabel('Frequency')
plt.xticks(rotation=45)
plt.tight_layout()
plt.show()

# Convert tokens to a single string to prepare for vectorization
data['processed_text'] = data['lemmatized_tokens'].apply(lambda x: ' '.join(x))


### DATA PREPARATION ###

