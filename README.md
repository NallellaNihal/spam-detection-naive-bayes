# Spam Detection with Naive Bayes

This project classifies text messages as **spam** or **ham** using a TF-IDF Vectorizer and Multinomial Naive Bayes classifier.

## Features

- Text preprocessing using TF-IDF
- Spam/ham classification
- Naive Bayes machine learning model
- Model training and evaluation
- Save and reuse trained model
- Predict custom messages from terminal

## Tech Stack

- Python
- Pandas
- Scikit-learn
- Joblib

## Project Structure

```bash
spam-detection-naive-bayes/
│
├── data/
│   └── spam_dataset.csv
│
├── src/
│   ├── train.py
│   └── predict.py
│
├── requirements.txt
├── README.md
└── .gitignore
