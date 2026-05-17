import pandas as pd
import joblib

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.pipeline import make_pipeline
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, accuracy_score


DATA_PATH = "data/spam_dataset.csv"
MODEL_PATH = "spam_model.pkl"


def train_model():
    df = pd.read_csv(DATA_PATH)

    X = df["message"]
    y = df["label"]

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.25,
        random_state=42,
        stratify=y
    )

    model = make_pipeline(
        TfidfVectorizer(stop_words="english"),
        MultinomialNB()
    )

    model.fit(X_train, y_train)

    y_pred = model.predict(X_test)

    print("\nSpam Detection Model Report")
    print("=" * 40)
    print("Accuracy:", accuracy_score(y_test, y_pred))
    print("\nClassification Report:\n")
    print(classification_report(y_test, y_pred))

    joblib.dump(model, MODEL_PATH)
    print(f"\nModel saved successfully as: {MODEL_PATH}")


if __name__ == "__main__":
    train_model()
