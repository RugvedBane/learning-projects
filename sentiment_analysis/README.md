# Sentiment Analysis — RNN vs LSTM

A learning project focused on building a basic **sentiment analysis model** using the IMDB movie reviews dataset and comparing a **SimpleRNN** model with an **LSTM** model using TensorFlow/Keras.

The goal of this project was to understand how recurrent neural networks work for text classification and how LSTMs compare with a basic RNN.

## Dataset

The project uses the **IMDB Movie Reviews dataset** provided through `keras.datasets`.

- Binary sentiment classification
- Positive and negative movie reviews
- Vocabulary limited to 10,000 words
- Reviews padded/truncated to a fixed length of 150 tokens

## Models

Two models were implemented and compared:

- **SimpleRNN**
- **LSTM**

Both models use an embedding layer followed by the respective recurrent layer for sentiment classification.

## What I Learned

- Loading and preprocessing text data with Keras
- Working with tokenized integer sequences and padding
- Using an `Embedding` layer
- Building a SimpleRNN model
- Building an LSTM model
- Training neural networks for binary classification
- Comparing RNN and LSTM performance

## Project Structure

```text
sentiment_analysis/
├── README.md
└── sentiment-analysis-rnn-lstm.ipynb