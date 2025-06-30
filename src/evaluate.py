import os
import json
import pandas as pd
import pickle
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from tensorflow import keras

def load_test_data(path="data/processed/test.csv"):
    df = pd.read_csv(path)
    df.columns = df.columns.str.strip()
    X = df.drop("income", axis=1)
    y = df["income"].astype(str).str.strip().apply(lambda s: 1 if s == ">50K" else 0)
    return X, y

def evaluate_rf(X, y, model_path="models/rf_model.pkl"):
    with open(model_path, "rb") as f:
        rf = pickle.load(f)
    preds = rf.predict(X)
    return {
        "accuracy":  accuracy_score(y, preds),
        "precision": precision_score(y, preds),
        "recall":    recall_score(y, preds),
        "f1":        f1_score(y, preds)
    }

def evaluate_nn(X, y, model_path="models/nn_model.h5"):
    model = keras.models.load_model(model_path)
    probs = model.predict(X)
    preds = (probs.flatten() >= 0.5).astype(int)
    return {
        "accuracy":  accuracy_score(y, preds),
        "precision": precision_score(y, preds),
        "recall":    recall_score(y, preds),
        "f1":        f1_score(y, preds)
    }

if __name__ == "__main__":
    # 1) Load test set
    X_test, y_test = load_test_data()

    # 2) Evaluate both models
    rf_m = evaluate_rf(X_test, y_test)
    nn_m = evaluate_nn(X_test, y_test)

    # 3) Write metrics.json with prefixed keys
    all_metrics = {
        "rf_accuracy":  rf_m["accuracy"],
        "rf_precision": rf_m["precision"],
        "rf_recall":    rf_m["recall"],
        "rf_f1":        rf_m["f1"],
        "nn_accuracy":  nn_m["accuracy"],
        "nn_precision": nn_m["precision"],
        "nn_recall":    nn_m["recall"],
        "nn_f1":        nn_m["f1"],
    }
    with open("metrics.json", "w") as f:
        json.dump(all_metrics, f, indent=4)

    # 4) Export metrics.csv for DVC plots
    plot_df = pd.DataFrame([
        {"model": "rf", **rf_m},
        {"model": "nn", **nn_m}
    ])
    plot_df.to_csv("metrics.csv", index=False)

    print("Evaluation complete. Saved metrics.json and metrics.csv")
