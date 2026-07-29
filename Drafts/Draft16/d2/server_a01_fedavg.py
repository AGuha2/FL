"""Pure FedAvg baseline for the alpha=0.1 equal-size seven-class split."""
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
from model import build_model


NUM_CLIENTS = 5
NUM_ROUNDS = 10
SEED = 42

SERVER_DATA_PATH = r"D:\CAPSTONE\Server_Test_7Class.csv"
SCALER_PATH = r"D:\CAPSTONE\global_scaler_7Class.pkl"
RESULT_DIR = Path("results_10round_7class_equal_a01_fedavg")
RESULT_DIR.mkdir(parents=True, exist_ok=True)

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
x_val, x_test, y_val, y_test = train_test_split(
    server_x,
    server_y,
    test_size=0.5,
    random_state=SEED,
    stratify=server_y,
)

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

    row = {"round": int(server_round), "test_loss": test_loss}
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
        f"[Alpha 0.1 FedAvg Server] R{server_round}: "
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


class MetricsFedAvg(fl.server.strategy.FedAvg):
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
                    f"Client {client_id}: invalid confusion matrix."
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


strategy = MetricsFedAvg(
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
    raise ValueError("Usage: python server_a01_fedavg.py <port>")

fl.server.start_server(
    server_address=f"localhost:{sys.argv[1]}",
    config=fl.server.ServerConfig(num_rounds=NUM_ROUNDS),
    strategy=strategy,
    grpc_max_message_length=1024 * 1024 * 1024,
)

print(
    "[Alpha 0.1 FedAvg Server] "
    f"Best validation macro-F1={best_val_f1:.4f} "
    f"at round {best_round}"
)
