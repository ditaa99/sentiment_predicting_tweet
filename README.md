# Twitter Sentiment Analysis

This project aims to classify Tweets as either positive or negative using NLP techniques. 
The analysis involves preprocessing raw tweet data, cleaning it, tokenizing, and performing sentiment classification using machine learning models.

## Table of Contents
- [Project Overview](#project-overview)
- [Data Collection](#data-collection)
- [Requirements](#requirements)
- [Usage](#usage)
  - [1. Importing the Dataset](#1-importing-the-dataset)
  - [2. Running the Main Script](#2-running-the-main-script)
- [Results](#results)
- [Model Evaluation](#model-evaluation)
- [Contributing](#contributing)
- [License](#license)

## Project Overview

The objective of this project is to analyze the sentiment of Tweets and classify them as positive or negative. The project encompasses data collection, preprocessing, feature extraction, model training, and evaluation.

## Data Collection

The dataset consists of 10,000 highly polar Tweets, with an equal distribution of positive and negative sentiments. These Tweets were collected from the Twitter Streaming APIs and provided in JSON format. The dataset files are:
- `negative_tweets.json`
- `positive_tweets.json`

### Requirements

1. **Python**:
   - Make sure you have Python 3 installed.

2. **Libraries**:
   - The following Python libraries are required for this project:
     - `pandas`
     - `matplotlib`
     - `nltk`
     - `scikit-learn`
     - `gensim`
     - `seaborn`

3. **NLTK Resources** are downloaded when running the main file, including:
     - `punkt`
     - `stopwords`
     - `wordnet`
     - `omw-1.4`
     - `twitter_samples`


## Usage

### 1. Importing the Dataset

First, run the `import_dataset.py` script to download and prepare the dataset.

Running this script will download the Twitter samples and print the number of positive and negative Tweets.

### 2. Running the Main Script

Next, run the `main.py` script where the source code for the sentiment analysis is located. This script includes the steps for preprocessing, feature extraction, model training, and evaluation:

```bash
python main.py
```

### Results

The script will output various metrics and visualizations, including:
- Accuracy, precision, recall of the trained model.
- Confusion matrix, ROC curve, and precision-recall curve to visualize the model's performance.

### Model Evaluation

The model's performance is evaluated using the following metrics:
- **Confusion Matrix**: Provides a detailed breakdown of the true positives, true negatives, false positives, and false negatives.
- **ROC Curve**: Plots the true positive rate against the false positive rate at various thresholds. The AUC score indicates the model's overall performance (closer to 1, better the performance).
- **Precision-Recall Curve**: Illustrates the trade-off between precision and recall, particularly useful for imbalanced datasets.

## Contributing

Contributions are welcome! If you have any suggestions or improvements, feel free to open a pull request or issue.

## License

This project is licensed under the MIT License.
