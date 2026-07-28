"""Flower client 4. Copy is intentionally standalone except for model/labels."""
import json
import random
import sys

import flwr as fl
import joblib
import numpy as np
import pandas as pd
import shap
import tensorflow as tf
from sklearn.metrics import (
    accuracy_score, balanced_accuracy_score, confusion_matrix,
    precision_recall_fscore_support,
)
from sklearn.model_selection import train_test_split
from sklearn.utils.class_weight import compute_class_weight
from tensorflow import keras

from Labels import ALL_LABELS, NUM_CLASSES, NUM_FEATURES, le
from model import build_model

CLIENT_ID = 4
DATA_PATH = rf"D:\CAPSTONE\Dir_Client{CLIENT_ID}.csv"
SCALER_PATH = r"D:\CAPSTONE\global_scaler_Class.pkl"
SEED = 42
LOCAL_EPOCHS = 1
BATCH_SIZE = 512
INITIAL_LR = 3e-4
LR_DECAY = 0.94
SHAP_BACKGROUND_SIZE = 25
SHAP_SAMPLES_PER_CLASS = 10
FINAL_SERVER_ROUND = 10
PERSONALIZATION_EPOCHS = 1
PERSONALIZATION_LR = 1e-5

random.seed(SEED + CLIENT_ID)
np.random.seed(SEED + CLIENT_ID)
tf.keras.utils.set_random_seed(SEED + CLIENT_ID)


def classification_metrics(y_true, y_pred):
    precision, recall, f1, _ = precision_recall_fscore_support(
        y_true, y_pred, average="macro", zero_division=0
    )
    matrix = confusion_matrix(y_true, y_pred, labels=np.arange(NUM_CLASSES))
    return {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "balanced_accuracy": float(balanced_accuracy_score(y_true, y_pred)),
        "precision": float(precision),
        "recall": float(recall),
        "f1": float(f1),
        "confusion_matrix": json.dumps(matrix.tolist()),
    }


def class_specific_mean_abs_shap(
    model,
    background,
    sample,
    sample_labels,
    number_of_features,
    number_of_classes,
):
    """Explain output c only on locally sampled records whose true class is c."""
    values = shap.GradientExplainer(model, background).shap_values(sample)
    result = {}

    for class_id in range(number_of_classes):
        class_mask = sample_labels == class_id
        if not np.any(class_mask):
            continue

        if isinstance(values, list):
            # Older SHAP: one (samples, features) array per model output.
            class_values = np.asarray(values[class_id])[class_mask, :]
        else:
            array = np.asarray(values)
            if (
                array.ndim == 3
                and array.shape[0] == len(sample)
                and array.shape[1] == number_of_features
                and array.shape[2] == number_of_classes
            ):
                # Newer SHAP: (samples, features, outputs).
                class_values = array[class_mask, :, class_id]
            elif (
                array.ndim == 3
                and array.shape[0] == number_of_classes
                and array.shape[1] == len(sample)
                and array.shape[2] == number_of_features
            ):
                # Alternative: (outputs, samples, features).
                class_values = array[class_id, class_mask, :]
            elif (
                array.ndim == 3
                and array.shape[0] == len(sample)
                and array.shape[1] == number_of_classes
                and array.shape[2] == number_of_features
            ):
                # Alternative: (samples, outputs, features).
                class_values = array[class_mask, class_id, :]
            else:
                raise ValueError(
                    f"Unexpected SHAP output shape: {array.shape}; "
                    f"expected {number_of_features} features and "
                    f"{number_of_classes} outputs."
                )

        class_values = np.asarray(class_values)
        if class_values.ndim == 3 and class_values.shape[-1] == 1:
            class_values = np.squeeze(class_values, axis=-1)

        vector = np.mean(np.abs(class_values), axis=0)
        if vector.shape != (number_of_features,):
            raise ValueError(
                f"Class {class_id} produced SHAP shape {vector.shape}, "
                f"expected ({number_of_features},)."
            )
        result[class_id] = np.asarray(vector, dtype=np.float64)

    return result


df = pd.read_csv(DATA_PATH).replace([np.inf, -np.inf], np.nan).dropna()
df.drop(columns=[c for c in df if c.lower().startswith("unnamed:")],
        inplace=True, errors="ignore")
feature_names = [c for c in df.columns if c != "Label"]
if len(feature_names) != NUM_FEATURES:
    raise ValueError(f"Client {CLIENT_ID}: expected {NUM_FEATURES} features, "
                     f"found {len(feature_names)}")

labels = df["Label"].astype(str).str.strip().str.upper()
unknown = sorted(set(labels) - set(ALL_LABELS))
if unknown:
    raise ValueError(f"Client {CLIENT_ID}: unexpected family labels {unknown}")

scaler = joblib.load(SCALER_PATH)
x = scaler.transform(df[feature_names]).astype(np.float32)
y = le.transform(labels).astype(np.int32)
try:
    x_train, x_test, y_train, y_test = train_test_split(
        x, y, test_size=0.2, random_state=SEED, stratify=y
    )
except ValueError:
    x_train, x_test, y_train, y_test = train_test_split(
        x, y, test_size=0.2, random_state=SEED
    )

present = np.unique(y_train)
raw_weights = compute_class_weight("balanced", classes=present, y=y_train)
raw_weights = np.clip(
    np.power(raw_weights, 0.75),
    0.5,
    10.0,
)
class_weights = dict(zip(present.astype(int), raw_weights.astype(float)))
class_counts = np.bincount(
    y_train,
    minlength=NUM_CLASSES,
).astype(np.int64)

model = build_model(NUM_FEATURES, NUM_CLASSES)
optimizer = keras.optimizers.AdamW(
    learning_rate=INITIAL_LR, weight_decay=1e-5, clipnorm=1.0
)
model.compile(optimizer=optimizer, loss="sparse_categorical_crossentropy")


def sparse_focal_loss(y_true, y_pred, gamma=2.0):
    """Return an unreduced focal loss for sparse integer labels."""
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
    correct_class_probability = tf.gather_nd(y_pred, indices)
    focal_factor = tf.pow(1.0 - correct_class_probability, gamma)
    return focal_factor * cross_entropy

def stratified_shap_indices(
    targets,
    samples_per_class=10,
    seed=42,
):
    """Select up to N records from every class available on this client."""
    local_rng = np.random.RandomState(seed)
    selected = []

    for class_id in range(NUM_CLASSES):
        class_indices = np.where(targets == class_id)[0]

        # Missing classes are normal for non-IID clients. They contribute
        # neither samples nor fabricated zero-valued class evidence.
        if len(class_indices) == 0:
            print(
                f"[Client {CLIENT_ID}] SHAP: "
                f"{ALL_LABELS[class_id]} unavailable locally; skipping."
            )
            continue

        number_to_select = min(samples_per_class, len(class_indices))
        selected.extend(
            local_rng.choice(
                class_indices,
                size=number_to_select,
                replace=False,
            ).tolist()
        )

    selected = np.asarray(selected, dtype=np.int64)
    local_rng.shuffle(selected)
    return selected


rng = np.random.RandomState(SEED + CLIENT_ID)
sample_idx = stratified_shap_indices(
    y_train,
    samples_per_class=SHAP_SAMPLES_PER_CLASS,
    seed=SEED + CLIENT_ID,
)
if len(sample_idx) == 0:
    raise ValueError(f"Client {CLIENT_ID} has no usable SHAP samples.")

# Keep background and explanation rows separate when enough data is available.
remaining = np.setdiff1d(np.arange(len(x_train)), sample_idx)
background_pool = remaining if len(remaining) else np.arange(len(x_train))
bg_idx = rng.choice(
    background_pool,
    size=min(SHAP_BACKGROUND_SIZE, len(background_pool)),
    replace=False,
)
shap_background = x_train[bg_idx]
shap_sample = x_train[sample_idx]
shap_sample_labels = y_train[sample_idx]
print(
    f"[Client {CLIENT_ID}] Stratified SHAP sample: "
    f"{len(shap_sample)} rows across "
    f"{len(np.unique(y_train))} available classes."
)


def train_epoch(global_trainable, proximal_mu):
    anchor = [tf.constant(weight) for weight in global_trainable]
    order = np.random.permutation(len(x_train))
    for start in range(0, len(order), BATCH_SIZE):
        idx = order[start:start + BATCH_SIZE]
        xb = tf.convert_to_tensor(x_train[idx], dtype=tf.float32)
        yb = tf.convert_to_tensor(y_train[idx], dtype=tf.int32)
        sw = tf.convert_to_tensor(
            [class_weights.get(int(label), 1.0) for label in y_train[idx]],
            dtype=tf.float32,
        )
        with tf.GradientTape() as tape:
            predictions = model(xb, training=True)
            per_sample_loss = sparse_focal_loss(
                yb,
                predictions,
                gamma=2.0,
            )
            ce = tf.reduce_sum(
                per_sample_loss * sw
            ) / (tf.reduce_sum(sw) + 1e-8)
            prox = tf.add_n([
                tf.reduce_sum(tf.square(local - global_))
                for local, global_ in zip(model.trainable_variables, anchor)
            ])
            loss = ce + 0.5 * proximal_mu * prox
        gradients = tape.gradient(loss, model.trainable_variables)
        gradients, _ = tf.clip_by_global_norm(gradients, 1.0)
        optimizer.apply_gradients(zip(gradients, model.trainable_variables))


def evaluate_local():
    probabilities = model.predict(x_test, batch_size=2048, verbose=0)
    predictions = probabilities.argmax(axis=1)
    loss = float(np.mean(
        keras.losses.sparse_categorical_crossentropy(y_test, probabilities)
    ))
    return loss, classification_metrics(y_test, predictions)


def personalize_output_layer():
    """Fine-tune only the classification head after the final global round."""
    for layer in model.layers:
        layer.trainable = False
    model.layers[-1].trainable = True

    personal_optimizer = keras.optimizers.Adam(
        learning_rate=PERSONALIZATION_LR,
        clipnorm=1.0,
    )

    for _ in range(PERSONALIZATION_EPOCHS):
        order = np.random.permutation(len(x_train))
        for start in range(0, len(order), BATCH_SIZE):
            idx = order[start:start + BATCH_SIZE]
            xb = tf.convert_to_tensor(x_train[idx], dtype=tf.float32)
            yb = tf.convert_to_tensor(y_train[idx], dtype=tf.int32)
            sw = tf.convert_to_tensor(
                [class_weights.get(int(label), 1.0) for label in y_train[idx]],
                dtype=tf.float32,
            )
            with tf.GradientTape() as tape:
                predictions = model(xb, training=True)
                per_sample_loss = sparse_focal_loss(
                    yb,
                    predictions,
                    gamma=2.0,
                )
                loss = tf.reduce_sum(
                    per_sample_loss * sw
                ) / (tf.reduce_sum(sw) + 1e-8)
            variables = model.trainable_variables
            gradients = tape.gradient(loss, variables)
            gradients, _ = tf.clip_by_global_norm(gradients, 1.0)
            personal_optimizer.apply_gradients(zip(gradients, variables))


class FlowerClient(fl.client.NumPyClient):
    def get_parameters(self, config):
        return model.get_weights()

    def fit(self, parameters, config):
        model.set_weights(parameters)
        round_number = int(config.get("server_round", 1))
        optimizer.learning_rate.assign(
            INITIAL_LR * LR_DECAY ** max(round_number - 1, 0)
        )
        global_trainable = [v.numpy().copy() for v in model.trainable_variables]
        for _ in range(LOCAL_EPOCHS):
            train_epoch(global_trainable, float(config.get("proximal_mu", 0.01)))

        local_loss, metrics = evaluate_local()
        class_shap_vectors = class_specific_mean_abs_shap(
            model,
            shap_background,
            shap_sample,
            shap_sample_labels,
            len(feature_names),
            NUM_CLASSES,
        )

        # Equal class averaging prevents large DDOS/DOS samples from erasing
        # rare-class explanations in the shared-layer SHAP summary.
        shap_vector = np.mean(
            np.stack(list(class_shap_vectors.values()), axis=0),
            axis=0,
        )
        fit_metrics = {
            "client_id": str(CLIENT_ID),
            "local_loss": local_loss,
            **{f"local_{k}": v for k, v in metrics.items()
               if k != "confusion_matrix"},
            **{f"shap::{name}": float(value)
               for name, value in zip(feature_names, shap_vector)},
            **{
                f"class_shap::{ALL_LABELS[class_id]}::{feature_name}":
                    float(class_vector[feature_index])
                for class_id, class_vector in class_shap_vectors.items()
                for feature_index, feature_name in enumerate(feature_names)
            },
            **{
                f"class_count::{class_name}": int(
                    class_counts[class_index]
                )
                for class_index, class_name
                in enumerate(ALL_LABELS)
            },
        }
        return model.get_weights(), len(x_train), fit_metrics

    def evaluate(self, parameters, config):
        model.set_weights(parameters)
        global_loss, global_metrics = evaluate_local()
        returned_metrics = {
            "client_id": str(CLIENT_ID),
            **global_metrics,
        }

        round_number = int(config.get("server_round", 0))
        if round_number == FINAL_SERVER_ROUND:
            personalize_output_layer()
            personalized_loss, personalized_metrics = evaluate_local()
            returned_metrics.update({
                "personalized_loss": personalized_loss,
                **{
                    f"personalized_{name}": value
                    for name, value in personalized_metrics.items()
                },
            })
            print(
                f"[Client {CLIENT_ID}] Personalized final metrics: "
                f"accuracy={personalized_metrics['accuracy']:.4f}, "
                f"F1={personalized_metrics['f1']:.4f}"
            )

        return global_loss, len(x_test), returned_metrics


if len(sys.argv) < 2:
    raise ValueError("Usage: python client4.py <port>")

fl.client.start_numpy_client(
    server_address=f"localhost:{sys.argv[1]}",
    client=FlowerClient(),
    grpc_max_message_length=1024 * 1024 * 1024,
)

