"""Seven-class FedAvg baseline client 2."""
import json
import random
import sys

import flwr as fl
import joblib
import numpy as np
import pandas as pd
import tensorflow as tf
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    confusion_matrix,
    precision_recall_fscore_support,
)
from sklearn.model_selection import train_test_split
from sklearn.utils.class_weight import compute_class_weight
from tensorflow import keras

from Labels7 import ALL_LABELS, NUM_CLASSES, NUM_FEATURES, le
from model import build_model

CLIENT_ID = 2
DATA_PATH = rf"D:\CAPSTONE\Dir7_Client{CLIENT_ID}.csv"
SCALER_PATH = r"D:\CAPSTONE\global_scaler_7Class.pkl"
SEED = 42
LOCAL_EPOCHS = 1
BATCH_SIZE = 512
INITIAL_LR = 3e-4
LR_DECAY = 0.94

random.seed(SEED + CLIENT_ID)
np.random.seed(SEED + CLIENT_ID)
tf.keras.utils.set_random_seed(SEED + CLIENT_ID)


def classification_metrics(y_true, y_pred):
    precision, recall, f1, _ = precision_recall_fscore_support(
        y_true,
        y_pred,
        average="macro",
        zero_division=0,
    )
    matrix = confusion_matrix(
        y_true,
        y_pred,
        labels=np.arange(NUM_CLASSES),
    )
    return {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "balanced_accuracy": float(
            balanced_accuracy_score(y_true, y_pred)
        ),
        "precision": float(precision),
        "recall": float(recall),
        "f1": float(f1),
        "confusion_matrix": json.dumps(matrix.tolist()),
    }


def sparse_focal_loss(y_true, y_pred, gamma=2.0):
    y_true = tf.cast(y_true, tf.int32)
    y_pred = tf.clip_by_value(
        y_pred,
        keras.backend.epsilon(),
        1.0 - keras.backend.epsilon(),
    )
    cross_entropy = keras.losses.sparse_categorical_crossentropy(
        y_true,
        y_pred,
    )
    indices = tf.stack(
        [tf.range(tf.shape(y_true)[0]), y_true],
        axis=1,
    )
    correct_probability = tf.gather_nd(y_pred, indices)
    return tf.pow(1.0 - correct_probability, gamma) * cross_entropy


df = pd.read_csv(DATA_PATH).replace(
    [np.inf, -np.inf],
    np.nan,
).dropna()
df.drop(
    columns=[
        column
        for column in df.columns
        if column.lower().startswith("unnamed:")
    ],
    inplace=True,
    errors="ignore",
)

feature_names = [
    column for column in df.columns
    if column != "Label"
]
if len(feature_names) != NUM_FEATURES:
    raise ValueError(
        f"Client {CLIENT_ID}: expected {NUM_FEATURES} features, "
        f"found {len(feature_names)}"
    )

labels = df["Label"].astype(str).str.strip().str.upper()
unknown = sorted(set(labels) - set(ALL_LABELS))
if unknown:
    raise ValueError(
        f"Client {CLIENT_ID}: unexpected labels {unknown}"
    )

scaler = joblib.load(SCALER_PATH)
x = scaler.transform(df[feature_names]).astype(np.float32)
y = le.transform(labels).astype(np.int32)

try:
    x_train, x_test, y_train, y_test = train_test_split(
        x,
        y,
        test_size=0.2,
        random_state=SEED,
        stratify=y,
    )
except ValueError:
    x_train, x_test, y_train, y_test = train_test_split(
        x,
        y,
        test_size=0.2,
        random_state=SEED,
    )

present_classes = np.unique(y_train)
raw_weights = compute_class_weight(
    "balanced",
    classes=present_classes,
    y=y_train,
)
raw_weights = np.clip(
    np.power(raw_weights, 0.75),
    0.5,
    10.0,
)
class_weights = dict(
    zip(
        present_classes.astype(int),
        raw_weights.astype(float),
    )
)

model = build_model(NUM_FEATURES, NUM_CLASSES)
optimizer = keras.optimizers.AdamW(
    learning_rate=INITIAL_LR,
    weight_decay=1e-5,
    clipnorm=1.0,
)
model.compile(
    optimizer=optimizer,
    loss="sparse_categorical_crossentropy",
)


def train_epoch():
    order = np.random.permutation(len(x_train))
    for start in range(0, len(order), BATCH_SIZE):
        indices = order[start:start + BATCH_SIZE]
        x_batch = tf.convert_to_tensor(
            x_train[indices],
            dtype=tf.float32,
        )
        y_batch = tf.convert_to_tensor(
            y_train[indices],
            dtype=tf.int32,
        )
        sample_weights = tf.convert_to_tensor(
            [
                class_weights.get(int(label), 1.0)
                for label in y_train[indices]
            ],
            dtype=tf.float32,
        )

        with tf.GradientTape() as tape:
            predictions = model(x_batch, training=True)
            per_sample_loss = sparse_focal_loss(
                y_batch,
                predictions,
                gamma=2.0,
            )
            loss = tf.reduce_sum(
                per_sample_loss * sample_weights
            ) / (tf.reduce_sum(sample_weights) + 1e-8)

        gradients = tape.gradient(
            loss,
            model.trainable_variables,
        )
        gradients, _ = tf.clip_by_global_norm(gradients, 1.0)
        optimizer.apply_gradients(
            zip(gradients, model.trainable_variables)
        )


def evaluate_local():
    probabilities = model.predict(
        x_test,
        batch_size=2048,
        verbose=0,
    )
    predictions = probabilities.argmax(axis=1)
    loss = float(np.mean(
        keras.losses.sparse_categorical_crossentropy(
            y_test,
            probabilities,
        )
    ))
    return loss, classification_metrics(y_test, predictions)


class FedAvgClient(fl.client.NumPyClient):
    def get_parameters(self, config):
        return model.get_weights()

    def fit(self, parameters, config):
        model.set_weights(parameters)
        round_number = int(config.get("server_round", 1))
        optimizer.learning_rate.assign(
            INITIAL_LR * LR_DECAY ** max(round_number - 1, 0)
        )
        for _ in range(LOCAL_EPOCHS):
            train_epoch()

        local_loss, local_metrics = evaluate_local()
        return model.get_weights(), len(x_train), {
            "client_id": str(CLIENT_ID),
            "local_loss": local_loss,
            **{
                f"local_{name}": value
                for name, value in local_metrics.items()
                if name != "confusion_matrix"
            },
        }

    def evaluate(self, parameters, config):
        model.set_weights(parameters)
        loss, result = evaluate_local()
        return loss, len(x_test), {
            "client_id": str(CLIENT_ID),
            **result,
        }


if len(sys.argv) < 2:
    raise ValueError("Usage: python client2.py <port>")

fl.client.start_numpy_client(
    server_address=f"localhost:{sys.argv[1]}",
    client=FedAvgClient(),
    grpc_max_message_length=1024 * 1024 * 1024,
)

