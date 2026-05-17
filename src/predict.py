import joblib


MODEL_PATH = "spam_model.pkl"


def predict_message(message):
    model = joblib.load(MODEL_PATH)
    prediction = model.predict([message])[0]
    probability = model.predict_proba([message])[0]

    classes = model.classes_
    confidence = max(probability)

    print("\nMessage:")
    print(message)

    print("\nPrediction:", prediction)
    print("Confidence:", round(confidence * 100, 2), "%")

    print("\nClass Probabilities:")
    for label, prob in zip(classes, probability):
        print(f"{label}: {round(prob * 100, 2)}%")


if __name__ == "__main__":
    msg = input("Enter a message to check: ")
    predict_message(msg)
