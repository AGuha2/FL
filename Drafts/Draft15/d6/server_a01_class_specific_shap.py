"""Alpha=0.1 seven-class server with class-specific SHAP aggregation."""
import json
import sys
from pathlib import Path

import flwr as fl
import joblib
import numpy as np
import pandas as pd
import tensorflow as tf
from flwr.common import ndarrays_to_parameters
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    confusion_matrix,
    precision_recall_fscore_support,
)
from sklearn.model_selection import train_test_split

from Labels7 import ALL_LABELS, NUM_CLASSES, NUM_FEATURES, le
from improved_shap_strategy import ClassSpecificReliableShap
from model import build_model


NUM_CLIENTS = 5
NUM_ROUNDS = 10
SEED = 42

SERVER_DATA_PATH = r"D:\CAPSTONE\Server_Test_7Class.csv"
SCALER_PATH = r"D:\CAPSTONE\global_scaler_7Class.pkl"
RESULT_DIR = Path(
    "results_10round_7class_equal_a01_class_specific_shap"
)
RESULT_DIR.mkdir(parents=True, exist_ok=True)

# Aggregation hyperparameters
WARMUP_ROUNDS = 2
SHAP_BLEND = 0.50
SHAP_TEMPERATURE = 0.25
MIN_CLIENT_WEIGHT = 0.12
MAX_CLIENT_WEIGHT = 0.28
WEIGHT_SMOOTHING = 0.70

# Balanced server subsets used for reliable model scoring and SHAP.
UTILITY_SAMPLES_PER_CLASS = 2000
SHAP_SAMPLES_PER_CLASS = 10
SHAP_BACKGROUND_SIZE = 25

np.random.seed(SEED)
tf.keras.utils.set_random_seed(SEED)


def calculate_metrics(y_true, y_pred):
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
        "confusion_matrix": matrix,
    }


def save_matrix(matrix, path):
    pd.DataFrame(
        matrix,
        index=[f"true_{name}" for name in ALL_LABELS],
        columns=[f"pred_{name}" for name in ALL_LABELS],
    ).to_csv(path)


def balanced_indices(targets, samples_per_class, seed):
    rng = np.random.RandomState(seed)
    selected = []
    for class_id in range(NUM_CLASSES):
        indices = np.where(targets == class_id)[0]
        if len(indices) == 0:
            raise ValueError(
                f"Server validation data has no {ALL_LABELS[class_id]}."
            )
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
# Load the separate server-side dataset.
# ---------------------------------------------------------------------------
server_df = pd.read_csv(SERVER_DATA_PATH).replace(
    [np.inf, -np.inf],
    np.nan,
).dropna()
server_df.drop(
    columns=[
        column
        for column in server_df.columns
        if column.lower().startswith("unnamed:")
    ],
    inplace=True,
    errors="ignore",
)

feature_names = [
    column for column in server_df.columns
    if column != "Label"
]
if len(feature_names) != NUM_FEATURES:
    raise ValueError(
        f"Expected {NUM_FEATURES} server features, "
        f"found {len(feature_names)}."
    )

labels = server_df["Label"].astype(str).str.strip().str.upper()
unknown = sorted(set(labels) - set(ALL_LABELS))
if unknown:
    raise ValueError(f"Unexpected server labels: {unknown}")

scaler = joblib.load(SCALER_PATH)
server_x = scaler.transform(
    server_df[feature_names]
).astype(np.float32)
server_y = le.transform(labels).astype(np.int32)

# Half of the held-out server data is validation; half is final testing.
x_val, x_test, y_val, y_test = train_test_split(
    server_x,
    server_y,
    test_size=0.5,
    random_state=SEED,
    stratify=server_y,
)

# Balanced validation subset for class-specific reliability.
utility_indices = balanced_indices(
    y_val,
    UTILITY_SAMPLES_PER_CLASS,
    SEED + 1000,
)
x_utility = x_val[utility_indices]
y_utility = y_val[utility_indices]

# Much smaller balanced subset for server SHAP calculation.
server_shap_indices = balanced_indices(
    y_val,
    SHAP_SAMPLES_PER_CLASS,
    SEED + 2000,
)
server_shap_samples = x_val[server_shap_indices]
server_shap_labels = y_val[server_shap_indices]

background_rng = np.random.RandomState(SEED + 3000)
background_pool = np.setdiff1d(
    np.arange(len(x_val)),
    server_shap_indices,
)
if len(background_pool) == 0:
    background_pool = np.arange(len(x_val))
server_background_indices = background_rng.choice(
    background_pool,
    size=min(SHAP_BACKGROUND_SIZE, len(background_pool)),
    replace=False,
)
server_shap_background = x_val[server_background_indices]


# ---------------------------------------------------------------------------
# Model and centralized evaluation.
# ---------------------------------------------------------------------------
evaluation_model = build_model(NUM_FEATURES, NUM_CLASSES)
evaluation_model.compile(loss="sparse_categorical_crossentropy")
initial_parameters = ndarrays_to_parameters(
    evaluation_model.get_weights()
)

server_history = []
client_history = []
best_val_f1 = -1.0
best_round = 0


def fit_config(server_round):
    return {"server_round": int(server_round)}


def evaluate_config(server_round):
    return {"server_round": int(server_round)}


def centralized_evaluate(server_round, parameters, config):
    global best_val_f1, best_round
    evaluation_model.set_weights(parameters)

    val_probabilities = evaluation_model.predict(
        x_val,
        batch_size=2048,
        verbose=0,
    )
    test_probabilities = evaluation_model.predict(
        x_test,
        batch_size=2048,
        verbose=0,
    )

    val_result = calculate_metrics(
        y_val,
        val_probabilities.argmax(axis=1),
    )
    test_result = calculate_metrics(
        y_test,
        test_probabilities.argmax(axis=1),
    )
    test_loss = float(np.mean(
        tf.keras.losses.sparse_categorical_crossentropy(
            y_test,
            test_probabilities,
        )
    ))

    row = {
        "round": int(server_round),
        "test_loss": test_loss,
    }
    for name in (
        "accuracy",
        "balanced_accuracy",
        "precision",
        "recall",
        "f1",
    ):
        row[f"val_{name}"] = val_result[name]
        row[f"test_{name}"] = test_result[name]

    server_history.append(row)
    pd.DataFrame(server_history).to_csv(
        RESULT_DIR / "server_metrics.csv",
        index=False,
    )
    save_matrix(
        test_result["confusion_matrix"],
        RESULT_DIR
        / f"server_confusion_round_{server_round:03d}.csv",
    )

    if server_round > 0 and val_result["f1"] > best_val_f1:
        best_val_f1 = val_result["f1"]
        best_round = int(server_round)
        evaluation_model.save_weights(
            RESULT_DIR / "best_global.weights.h5"
        )

    print(
        f"[Class-specific SHAP Server] R{server_round}: "
        f"accuracy={test_result['accuracy']:.4f}, "
        f"balanced_accuracy="
        f"{test_result['balanced_accuracy']:.4f}, "
        f"F1={test_result['f1']:.4f}"
    )
    return test_loss, {
        name: test_result[name]
        for name in (
            "accuracy",
            "balanced_accuracy",
            "precision",
            "recall",
            "f1",
        )
    }


class ExperimentStrategy(ClassSpecificReliableShap):
    """Adds per-client CSV metrics to class-specific SHAP aggregation."""

    def aggregate_evaluate(self, server_round, results, failures):
        if not results:
            return None, {}

        seen_ids = set()
        for _, evaluate_result in results:
            returned = dict(evaluate_result.metrics)
            client_id = int(returned["client_id"])
            if client_id in seen_ids:
                raise ValueError(
                    f"Duplicate evaluation client ID: {client_id}"
                )
            seen_ids.add(client_id)

            matrix = np.asarray(
                json.loads(returned["confusion_matrix"]),
                dtype=int,
            )
            if matrix.shape != (NUM_CLASSES, NUM_CLASSES):
                raise ValueError(
                    f"Client {client_id}: invalid confusion matrix "
                    f"shape {matrix.shape}."
                )
            save_matrix(
                matrix,
                RESULT_DIR
                / (
                    f"client_{client_id}_confusion_"
                    f"round_{server_round:03d}.csv"
                ),
            )

            client_history.append({
                "round": int(server_round),
                "client_id": client_id,
                "num_examples": int(evaluate_result.num_examples),
                "loss": float(evaluate_result.loss),
                **{
                    name: float(returned.get(name, 0.0))
                    for name in (
                        "accuracy",
                        "balanced_accuracy",
                        "precision",
                        "recall",
                        "f1",
                    )
                },
            })

        client_history.sort(
            key=lambda row: (row["round"], row["client_id"])
        )
        pd.DataFrame(client_history).to_csv(
            RESULT_DIR / "client_metrics.csv",
            index=False,
        )
        return super().aggregate_evaluate(
            server_round,
            results,
            failures,
        )


strategy = ExperimentStrategy(
    evaluation_model=evaluation_model,
    x_utility=x_utility,
    y_utility=y_utility,
    shap_background=server_shap_background,
    shap_samples=server_shap_samples,
    shap_labels=server_shap_labels,
    feature_names=feature_names,
    class_names=ALL_LABELS,
    result_dir=RESULT_DIR,
    warmup_rounds=WARMUP_ROUNDS,
    shap_blend=SHAP_BLEND,
    temperature=SHAP_TEMPERATURE,
    minimum_weight=MIN_CLIENT_WEIGHT,
    maximum_weight=MAX_CLIENT_WEIGHT,
    smoothing=WEIGHT_SMOOTHING,
    fraction_fit=1.0,
    fraction_evaluate=1.0,
    min_fit_clients=NUM_CLIENTS,
    min_evaluate_clients=NUM_CLIENTS,
    min_available_clients=NUM_CLIENTS,
    initial_parameters=initial_parameters,
    on_fit_config_fn=fit_config,
    on_evaluate_config_fn=evaluate_config,
    evaluate_fn=centralized_evaluate,
)


if len(sys.argv) < 2:
    raise ValueError("Usage: python server.py <port>")

fl.server.start_server(
    server_address=f"localhost:{sys.argv[1]}",
    config=fl.server.ServerConfig(num_rounds=NUM_ROUNDS),
    strategy=strategy,
    grpc_max_message_length=1024 * 1024 * 1024,
)

print(
    "[Class-specific SHAP Server] "
    f"Best validation macro-F1={best_val_f1:.4f} "
    f"at round {best_round}"
)
