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
from sklearn.feature_extraction.text import TfidfVectorizer
from gensim.models import Word2Vec
from sklearn.model_selection import train_test_split
# Train
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report
# Evaluate
import seaborn as sns
from sklearn.metrics import confusion_matrix
from sklearn.metrics import roc_curve, roc_auc_score
from sklearn.metrics import precision_recall_curve

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
negative_tweets = pd.read_json('twitter_samples/negative_tweets.json', lines=True)
positive_tweets = pd.read_json('twitter_samples/positive_tweets.json', lines=True)

#label the data
negative_tweets['label'] = 0
positive_tweets['label'] = 1

# Merge the dataframes
data = pd.concat([negative_tweets, positive_tweets], ignore_index=True)

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

# common tokens
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

# Feature extraction using TF-IDF
vectorizer = TfidfVectorizer(max_features=5000)
X = vectorizer.fit_transform(data['processed_text'])
y = data['label']

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Print shapes of the splits to confirm
print("Training set size:", X_train.shape)
print("Testing set size:", X_test.shape)


### MODEL TRAINING ###

# Train the model
model = LogisticRegression()
model.fit(X_train, y_train)

# Make predictions
y_pred = model.predict(X_test)

# Evaluate the model
print(f'Accuracy: {accuracy_score(y_test, y_pred)}')
print(classification_report(y_test, y_pred))

# Example predictions
new_tweets = ["I love this!", "This is terrible."]
new_tweets_processed = [clean_text(tweet) for tweet in new_tweets]
new_tweets_vec = vectorizer.transform(new_tweets_processed)
predictions = model.predict(new_tweets_vec)
print("Predictions:", predictions)


### PERFORMANCE OF OUR MODEL ###

# Confusion matrix
cm = confusion_matrix(y_test, y_pred)

# Plot confusion matrix
plt.figure(figsize=(8, 6))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
plt.xlabel('Predicted')
plt.ylabel('Actual')
plt.title('Confusion Matrix')
plt.show()


# ROC curve and AUC
y_pred_prob = model.predict_proba(X_test)[:, 1]
fpr, tpr, thresholds = roc_curve(y_test, y_pred_prob)
auc = roc_auc_score(y_test, y_pred_prob)

# Plot ROC curve
plt.figure(figsize=(8, 6))
plt.plot(fpr, tpr, label=f'AUC = {auc:.2f}')
plt.plot([0, 1], [0, 1], 'k--')
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('ROC Curve')
plt.legend(loc='lower right')
plt.show()


# Precision-recall curve
precision, recall, _ = precision_recall_curve(y_test, y_pred_prob)

# Plot precision-recall curve
plt.figure(figsize=(8, 6))
plt.plot(recall, precision)
plt.xlabel('Recall')
plt.ylabel('Precision')
plt.title('Precision-Recall Curve')
plt.show()
