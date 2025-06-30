import json
import pandas as pd
import pickle
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from tensorflow import keras

def load_test_data(path="data/processed/test.csv"):
    df = pd.read_csv(path)
    X = df.drop("income", axis=1)
    y = df["income"].apply(lambda s: 1 if s.strip() == ">50K" else 0)
    return X, y

def evaluate_rf(X, y, model_path="models/rf_model.pkl"):
    with open(model_path, "rb") as f:
        rf = pickle.load(f)
    preds = rf.predict(X)
    return {
        "rf_accuracy": accuracy_score(y, preds),
        "rf_precision": precision_score(y, preds),
        "rf_recall": recall_score(y, preds),
        "rf_f1": f1_score(y, preds)
    }

def evaluate_nn(X, y, model_path="models/nn_model.h5"):
    model = keras.models.load_model(model_path)
    probs = model.predict(X)
    preds = (probs.flatten() >= 0.5).astype(int)
    return {
        "nn_accuracy": accuracy_score(y, preds),
        "nn_precision": precision_score(y, preds),
        "nn_recall": recall_score(y, preds),
        "nn_f1": f1_score(y, preds)
    }

if __name__ == "__main__":
    # 1) Load test data
    X_test, y_test = load_test_data()

    # 2) Evaluate both models
    rf_metrics = evaluate_rf(X_test, y_test)
    nn_metrics = evaluate_nn(X_test, y_test)

    # 3) Combine and save
    all_metrics = {**rf_metrics, **nn_metrics}
    with open("metrics.json", "w") as f:
        json.dump(all_metrics, f, indent=4)

    print("Evaluation complete. Metrics:")
    for k, v in all_metrics.items():
        print(f"  {k}: {v:.4f}")
