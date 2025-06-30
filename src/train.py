import os
import argparse
import yaml
import json
import pandas as pd
import mlflow
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
import tensorflow as tf
from tensorflow import keras

def load_params(path="params.yaml"):
    with open(path, "r") as f:
        return yaml.safe_load(f)

def train_rf(params):
    # --- Load data ---
    df = pd.read_csv("data/processed/train.csv")
    X = df.drop("income", axis=1)
    y = df["income"].apply(lambda s: 1 if s.strip() == ">50K" else 0)

    # --- Start MLflow run ---
    mlflow.start_run()
    mlflow.log_params({
        "model": "RandomForest",
        "n_estimators": params["rf"]["n_estimators"],
        "max_depth": params["rf"]["max_depth"]
    })

    # --- Train model ---
    rf = RandomForestClassifier(
        n_estimators=params["rf"]["n_estimators"],
        max_depth=params["rf"]["max_depth"],
        random_state=42
    )
    rf.fit(X, y)

    # --- Evaluate on train set ---
    preds = rf.predict(X)
    acc = accuracy_score(y, preds)
    mlflow.log_metric("train_accuracy", acc)

    # --- Save model artifact ---
    os.makedirs("models", exist_ok=True)
    model_path = "models/rf_model.pkl"
    pd.to_pickle(rf, model_path)
    mlflow.log_artifact(model_path)

    # --- Write metrics JSON for DVC ---
    metrics = {"train_accuracy": float(acc)}
    with open("metrics_rf.json", "w") as f:
        json.dump(metrics, f, indent=4)

    mlflow.end_run()
    print(f"[RF] Done — train_accuracy={acc:.4f}")

def train_nn(params):
    # --- Load data ---
    df = pd.read_csv("data/processed/train.csv")
    X = df.drop("income", axis=1).values.astype("float32")
    y = df["income"].apply(lambda s: 1 if s.strip() == ">50K" else 0).values

    # --- Build model ---
    model = keras.Sequential([
        keras.layers.InputLayer(input_shape=(X.shape[1],)),
        keras.layers.Dense(64, activation="relu"),
        keras.layers.Dense(1, activation="sigmoid")
    ])
    optimizer = keras.optimizers.Adam(learning_rate=params["nn"]["learning_rate"])
    loss_fn = keras.losses.BinaryCrossentropy()
    model.compile(optimizer=optimizer, loss=loss_fn, metrics=["accuracy"])

    # --- Prepare TensorBoard logging ---
    logdir = "logs/nn"
    os.makedirs(logdir, exist_ok=True)
    writer = tf.summary.create_file_writer(logdir)

    # --- Start MLflow run ---
    mlflow.start_run()
    mlflow.log_params({
        "model": "NeuralNet",
        "learning_rate": params["nn"]["learning_rate"],
        "batch_size": params["nn"]["batch_size"],
        "epochs": params["nn"]["epochs"]
    })

    # --- Manual training loop with tf.summary ---
    batch_size = params["nn"]["batch_size"]
    epochs     = params["nn"]["epochs"]
    dataset    = tf.data.Dataset.from_tensor_slices((X, y)).batch(batch_size)

    for epoch in range(epochs):
        epoch_loss = tf.keras.metrics.Mean()
        epoch_acc  = tf.keras.metrics.BinaryAccuracy()

        for x_batch, y_batch in dataset:
            with tf.GradientTape() as tape:
                y_pred = model(x_batch, training=True)
                loss   = loss_fn(y_batch, y_pred)
            grads = tape.gradient(loss, model.trainable_variables)
            optimizer.apply_gradients(zip(grads, model.trainable_variables))

            epoch_loss.update_state(loss)
            epoch_acc.update_state(y_batch, y_pred)

        # log to TensorBoard
        with writer.as_default():
            tf.summary.scalar("loss", epoch_loss.result(), step=epoch)
            tf.summary.scalar("accuracy", epoch_acc.result(), step=epoch)

        print(f"[NN] Epoch {epoch+1}/{epochs} — loss={epoch_loss.result():.4f}, acc={epoch_acc.result():.4f}")

    # --- Final evaluation & MLflow logging ---
    final_loss, final_acc = model.evaluate(X, y, verbose=0)
    mlflow.log_metric("train_loss", float(final_loss))
    mlflow.log_metric("train_accuracy", float(final_acc))

    # --- Save model artifact ---
    nn_path = "models/nn_model.h5"
    model.save(nn_path)
    mlflow.log_artifact(nn_path)

    # --- Write metrics JSON for DVC ---
    metrics = {
        "train_loss": float(final_loss),
        "train_accuracy": float(final_acc)
    }
    with open("metrics_nn.json", "w") as f:
        json.dump(metrics, f, indent=4)

    mlflow.end_run()
    print(f"[NN] Done — train_accuracy={final_acc:.4f}, logs in '{logdir}'")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--stage", type=str, required=True,
        choices=["train_rf", "train_nn"]
    )
    args = parser.parse_args()

    params = load_params()
    if args.stage == "train_rf":
        train_rf(params)
    else:
        train_nn(params)
