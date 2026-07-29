"""Seven-class class-specific SHAP federated client 1."""
import json
import random
import sys
from pathlib import Path

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

from class_specific_shap_weighting import (
    class_specific_mean_abs_shap,
)
from Labels7 import ALL_LABELS, NUM_CLASSES, NUM_FEATURES, le
from model import build_model
from posthoc_shap_analysis import save_local_client_posthoc_shap


# Change only this value in client2.py ... client5.py.
CLIENT_ID = 4

DATA_PATH = rf"D:\CAPSTONE\Dir7Equal_Client{CLIENT_ID}.csv"
SCALER_PATH = r"D:\CAPSTONE\global_scaler_7Class.pkl"
LOCAL_POSTHOC_DIR = (
    Path("client_posthoc_results")
    / "Reliable_Consensus_SHAP"
    / f"client_{CLIENT_ID}"
)

SEED = 42
LOCAL_EPOCHS = 1
BATCH_SIZE = 512
INITIAL_LR = 3e-4
LR_DECAY = 0.94
SHAP_BACKGROUND_SIZE = 25
SHAP_SAMPLES_PER_CLASS = 10
POSTHOC_SHAP_BACKGROUND_SIZE = 50
POSTHOC_SHAP_SAMPLES_PER_CLASS = 50

random.seed(SEED + CLIENT_ID)
np.random.seed(SEED + CLIENT_ID)
tf.keras.utils.set_random_seed(SEED + CLIENT_ID)


def classification_metrics(y_true, y_pred):
    precision, recall, f1, _ = precision_recall_fscore_support(
        y_true,
        y_pred,
        labels=np.arange(NUM_CLASSES),
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
    return (
        tf.pow(1.0 - correct_probability, gamma)
        * cross_entropy
    )


def stratified_indices(targets, samples_per_class, seed):
    """Select at most N local examples from every available class."""
    rng = np.random.RandomState(seed)
    selected = []
    for class_id in range(NUM_CLASSES):
        indices = np.where(targets == class_id)[0]
        if len(indices) == 0:
            # An unavailable class remains unavailable; no fake samples.
            continue
        chosen = rng.choice(
            indices,
            size=min(samples_per_class, len(indices)),
            replace=False,
        )
        selected.extend(chosen.tolist())

    selected = np.asarray(selected, dtype=np.int64)
    rng.shuffle(selected)
    return selected


# ---------------------------------------------------------------------------
# Load this client's own dataset.
# ---------------------------------------------------------------------------
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
        f"found {len(feature_names)}."
    )

labels = df["Label"].astype(str).str.strip().str.upper()
unknown = sorted(set(labels) - set(ALL_LABELS))
if unknown:
    raise ValueError(
        f"Client {CLIENT_ID}: unexpected labels {unknown}."
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
    # This is only a fallback for a client with too few rare samples.
    x_train, x_test, y_train, y_test = train_test_split(
        x,
        y,
        test_size=0.2,
        random_state=SEED,
    )


# ---------------------------------------------------------------------------
# Local class-balanced focal-loss training.
# ---------------------------------------------------------------------------
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
class_weights = dict(zip(
    present_classes.astype(int),
    raw_weights.astype(float),
))

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


# ---------------------------------------------------------------------------
# Fixed, stratified local SHAP subset.
# ---------------------------------------------------------------------------
shap_sample_indices = stratified_indices(
    y_train,
    SHAP_SAMPLES_PER_CLASS,
    SEED + CLIENT_ID,
)
if len(shap_sample_indices) == 0:
    raise ValueError(f"Client {CLIENT_ID} has no SHAP samples.")

shap_sample = x_train[shap_sample_indices]
shap_sample_labels = y_train[shap_sample_indices]

shap_rng = np.random.RandomState(SEED + CLIENT_ID)
background_pool = np.setdiff1d(
    np.arange(len(x_train)),
    shap_sample_indices,
)
if len(background_pool) == 0:
    background_pool = np.arange(len(x_train))
background_indices = shap_rng.choice(
    background_pool,
    size=min(SHAP_BACKGROUND_SIZE, len(background_pool)),
    replace=False,
)
shap_background = x_train[background_indices]

# A larger, final-round-only sample from the local test partition is used
# for post-hoc explanation. It is never used for local training or weighting.
posthoc_sample_indices = stratified_indices(
    y_test,
    POSTHOC_SHAP_SAMPLES_PER_CLASS,
    SEED + 10_000 + CLIENT_ID,
)
posthoc_sample = x_test[posthoc_sample_indices]
posthoc_sample_labels = y_test[posthoc_sample_indices]
posthoc_rng = np.random.RandomState(SEED + 20_000 + CLIENT_ID)
posthoc_background_pool = np.setdiff1d(
    np.arange(len(x_test)),
    posthoc_sample_indices,
)
if len(posthoc_background_pool) == 0:
    posthoc_background_pool = np.arange(len(x_test))
posthoc_background_indices = posthoc_rng.choice(
    posthoc_background_pool,
    size=min(
        POSTHOC_SHAP_BACKGROUND_SIZE,
        len(posthoc_background_pool),
    ),
    replace=False,
)
posthoc_background = x_test[posthoc_background_indices]


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


def make_shap_metrics():
    """Flower metrics must be scalar, so flatten class-feature SHAP."""
    matrix, available = class_specific_mean_abs_shap(
        model=model,
        background=shap_background,
        samples=shap_sample,
        sample_labels=shap_sample_labels,
        num_classes=NUM_CLASSES,
        num_features=len(feature_names),
    )

    metrics = {}
    for class_id, class_name in enumerate(ALL_LABELS):
        metrics[f"shap_available::{class_name}"] = float(
            available[class_id]
        )
        for feature_id, feature_name in enumerate(feature_names):
            metrics[
                f"shap::{class_name}::{feature_name}"
            ] = float(matrix[class_id, feature_id])
    return metrics, matrix, available


def make_posthoc_shap_metrics():
    matrix, available = class_specific_mean_abs_shap(
        model=model,
        background=posthoc_background,
        samples=posthoc_sample,
        sample_labels=posthoc_sample_labels,
        num_classes=NUM_CLASSES,
        num_features=len(feature_names),
    )
    metrics = {}
    for class_id, class_name in enumerate(ALL_LABELS):
        metrics[f"posthoc_shap_available::{class_name}"] = float(
            available[class_id]
        )
        for feature_id, feature_name in enumerate(feature_names):
            metrics[
                f"posthoc_shap::{class_name}::{feature_name}"
            ] = float(matrix[class_id, feature_id])
    return metrics, matrix, available


class ClassSpecificShapClient(fl.client.NumPyClient):
    def get_parameters(self, config):
        return model.get_weights()

    def fit(self, parameters, config):
        model.set_weights(parameters)
        round_number = int(config.get("server_round", 1))
        optimizer.learning_rate.assign(
            INITIAL_LR
            * LR_DECAY ** max(round_number - 1, 0)
        )

        for _ in range(LOCAL_EPOCHS):
            train_epoch()

        local_loss, local_metrics = evaluate_local()
        shap_metrics, shap_matrix, shap_available = make_shap_metrics()
        posthoc_metrics = {}

        if bool(config.get("run_posthoc_shap", False)):
            (
                posthoc_metrics,
                posthoc_matrix,
                posthoc_available,
            ) = make_posthoc_shap_metrics()
            shap_path = save_local_client_posthoc_shap(
                LOCAL_POSTHOC_DIR,
                server_round=round_number,
                client_id=CLIENT_ID,
                class_names=ALL_LABELS,
                feature_names=feature_names,
                client_shap=posthoc_matrix,
                available=posthoc_available,
            )
            model.save_weights(
                LOCAL_POSTHOC_DIR
                / f"final_client_{CLIENT_ID}.weights.h5"
            )
            print(
                f"[Client {CLIENT_ID}] Final post-hoc SHAP saved to "
                f"{shap_path}"
            )

        return model.get_weights(), len(x_train), {
            "client_id": str(CLIENT_ID),
            "local_loss": float(local_loss),
            **{
                f"local_{name}": float(value)
                for name, value in local_metrics.items()
                if name != "confusion_matrix"
            },
            **shap_metrics,
            **posthoc_metrics,
        }

    def evaluate(self, parameters, config):
        model.set_weights(parameters)
        loss, result = evaluate_local()
        return loss, len(x_test), {
            "client_id": str(CLIENT_ID),
            **result,
        }


if len(sys.argv) < 2:
    raise ValueError("Usage: python client4.py <port>")

fl.client.start_numpy_client(
    server_address=f"localhost:{sys.argv[1]}",
    client=ClassSpecificShapClient(),
    grpc_max_message_length=1024 * 1024 * 1024,
)
