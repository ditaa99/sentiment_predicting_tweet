'''
Project A: Predicting Sentiment from Tweets

The problem is to determine whether a given Tweet has a positive or negative sentiment. First, you will 
need to perform data cleaning by removing punctuation, removing stop words and performing 
stemming. Next, you will split the dataset between training and testing. Finally, you will have to classify 
correctly the tweets of the testing set in either positive or negative.

This dataset contains 10,000 highly polar Tweets (50% positive and 50% negative). 
'''
import pandas as pd
import matplotlib.pyplot as plt
import nltk
from nltk.corpus import stopwords
import re

nltk.download('stopwords')  # NLTK stopwords
nltk.download('punkt_tab')      # punkt tokenizer
stop_words = set(stopwords.words('english'))

# Load and Merge the JSON Files
json_files = ['twitter_samples/negative_tweets.json', 'twitter_samples/positive_tweets.json']

# Merge into a single DataFrame
dataframes = [pd.read_json(file, lines=True) for file in json_files]
data = pd.concat(dataframes, ignore_index=True)

# Inspect Sample Tweets - see the structure of the data
# print("Sample Tweets:\n", data[['id_str', 'created_at', 'text']].head())

# Distribution of Tweet Lengths
# Calculate tweet lengths
data['text_length'] = data['text'].apply(lambda x: len(x) if pd.notnull(x) else 0)

# Plot distribution
plt.figure(figsize=(10, 5))
plt.hist(data['text_length'], bins=30, color='skyblue', edgecolor='black')
plt.title('Distribution of Tweet Lengths')
plt.xlabel('Tweet Length')
plt.ylabel('Frequency')
plt.show()

# Funct for Cleaning Text
def clean_text(text):
    text = re.sub(r'http\S+|www\S+|https\S+', '', text, flags=re.MULTILINE) 
    text = re.sub(r'\@\w+|\#', '', text)  # Remove mentions and hashtags
    text = re.sub(r'[^a-zA-Z\s]', '', text)  # Keep only letters and spaces
    text = text.lower()  # Convert to lowercase
    return text

# Clean the tweets
data['clean_text'] = data['text'].apply(lambda x: clean_text(x) if pd.notnull(x) else '')

# TOKENIZATION
data['tokens'] = data['clean_text'].apply(nltk.word_tokenize)

# Remove stopwords from tokens
data['tokens'] = data['tokens'].apply(lambda x: [word for word in x if word not in stop_words])

# Print sample tokenized text
print(data[['clean_text', 'tokens']].head())

# Get token frequency distribution
all_tokens = [token for sublist in data['tokens'] for token in sublist]
token_freq = nltk.FreqDist(all_tokens)

# Extract the most common tokens and their frequencies
most_common_tokens = token_freq.most_common(10)

# Prepare data for plotting
tokens, counts = zip(*most_common_tokens)

# Plot the most common tokens
plt.figure(figsize=(10, 6))
plt.bar(tokens, counts, color='skyblue', edgecolor='black')
plt.title('Most Common Tokens in Tweets')
plt.xlabel('Tokens')
plt.ylabel('Frequency')
plt.xticks(rotation=45)
plt.show()
