# Next Word Predictor using LSTM

A next word prediction model built with LSTM in Keras, trained on the Tiny Shakespeare dataset.

## About
This project converts raw text into a supervised multiclass classification problem where the model predicts the next word given a sequence of previous words.

## Dataset
- Tiny Shakespeare (~1.1M characters)
- Vocab size: 12,633 unique words

## Model Architecture
- Embedding Layer (12633, 150)
- LSTM (150 units)
- Dropout (0.2)
- Dense + Softmax (12633 units)

## Results
- Accuracy: 18.39% after 20 epochs
- Loss: 4.40

## What I Learned
- Text tokenization and sequence generation
- Converting NLP into a supervised classification problem using n-grams
- Padding sequences for LSTM input
- One-hot encoding for multiclass classification
- Training LSTMs in Keras

## Tech Stack
- Python
- TensorFlow / Keras
- NumPy
- Kaggle (GPU: Tesla P100)