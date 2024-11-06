# Amazon Reviews Sentiment Analysis with RNN and LSTM

This project performs sentiment analysis on Amazon product reviews, aiming to classify review scores using a Recurrent Neural Network (RNN) and Long Short-Term Memory (LSTM) model. We use natural language processing (NLP) techniques for data preprocessing and visualization, followed by deep learning techniques to build and evaluate our models.

## Project Overview

1. **Data Loading and Exploration**:
   - We load a sample of 30,000 Amazon reviews, then explore the data through visualization.
   - The `Score` distribution is visualized to understand the rating patterns.

2. **Data Cleaning**:
   - Text preprocessing steps are applied, including:
     - Lowercasing text, removing punctuation, HTML tags, and links.
     - Removing stopwords.
   - A Word Cloud is generated to visualize the most frequent words in positive and negative reviews.
   - We also analyze the distribution of word counts in 1-star and 5-star reviews.

3. **N-gram Analysis**:
   - We perform unigram, bigram, and trigram analysis to find the most common word sequences in the reviews.

4. **Tokenization and Embedding**:
   - Tokenize the cleaned text and pad sequences to ensure uniform input length.
   - Use Google’s pre-trained `word2vec` embeddings to create a matrix for our vocabulary.

5. **Model Implementation**:
   - **Simple RNN**:
     - A simple RNN model is built with an embedding layer, an RNN layer, dropout, and a dense output layer.
   - **LSTM**:
     - We use an LSTM model with stacked LSTM layers for enhanced performance on sequential data.

6. **Model Training and Evaluation**:
   - Both models are compiled and trained using `sparse_categorical_crossentropy` loss and the `adam` optimizer.
   - Training and validation loss and accuracy are plotted for each model.

## Project Structure

```plaintext
├── amazon_reviews.csv
├── README.md
├── amazon_reviews_analysis.py
├── requirements.txt
└── images
    ├── score_distribution.png
    ├── wordcloud_positive.png
    ├── wordcloud_negative.png
    ├── rnn_loss_accuracy.png
    └── lstm_loss_accuracy.png
