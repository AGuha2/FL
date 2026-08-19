import json
import sys
from pathlib import Path

import flwr as fl
import joblib
import numpy as np
import pandas as pd
import shap
import tensorflow as tf
from flwr.common import (
    ndarrays_to_parameters,
    parameters_to_ndarrays,
)
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    confusion_matrix,
    precision_recall_fscore_support,
)
from sklearn.model_selection import train_test_split

from Labels6 import ALL_LABELS, NUM_CLASSES, NUM_FEATURES, le
from model import build_model

NUM_CLIENTS = 5
NUM_ROUNDS = 10
SEED = 42
SHAP_BLEND = 0
SHAP_BACKGROUND_SIZE = 25
SHAP_SAMPLES_PER_CLASS = 10
SERVER_DATA_PATH = r"D:\CAPSTONE\IoT6Bal_Server.csv"
SCALER_PATH = r"D:\CAPSTONE\IoT6Bal_Scaler.pkl"
RESULT_DIR = Path(
    f"results_6class_bal_lstm_shapServer_{int(SHAP_BLEND * 100):02d}"
)
RESULT_DIR.mkdir(parents=True, exist_ok=True)

np.random.seed(SEED)
tf.keras.utils.set_random_seed(SEED)


def calculate_metrics(y_true, y_pred): # computes the macro metrics
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
        "confusion_matrix": matrix,
    }


def save_matrix(matrix, path): # writes and saves the confusion matrics
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
        f"found {len(feature_names)}"
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

# Fixed server-test samples used only for round-10 global SHAP.
server_shap_rng = np.random.RandomState(SEED + 1000)
server_shap_sample_indices = []
for class_id in range(NUM_CLASSES):
    class_indices = np.where(y_test == class_id)[0]
    if len(class_indices) == 0:
        continue
    server_shap_sample_indices.extend(
        server_shap_rng.choice(
            class_indices,
            size=min(SHAP_SAMPLES_PER_CLASS, len(class_indices)),
            replace=False,
        ).tolist()
    )
server_shap_sample_indices = np.asarray(
    server_shap_sample_indices,
    dtype=np.int64,
)
server_shap_rng.shuffle(server_shap_sample_indices)

server_background_pool = np.setdiff1d(
    np.arange(len(x_test)),
    server_shap_sample_indices,
)
if len(server_background_pool) == 0:
    server_background_pool = np.arange(len(x_test))
server_background_indices = server_shap_rng.choice(
    server_background_pool,
    size=min(SHAP_BACKGROUND_SIZE, len(server_background_pool)),
    replace=False,
)
server_shap_sample = x_test[server_shap_sample_indices]
server_shap_background = x_test[server_background_indices]

evaluation_model = build_model(NUM_FEATURES, NUM_CLASSES)
evaluation_model.compile(loss="sparse_categorical_crossentropy")
initial_parameters = ndarrays_to_parameters(
    evaluation_model.get_weights()
)

server_history = []
client_history = []
aggregation_history = []
best_val_f1 = -1.0
best_round = 0


def fit_config(server_round): # for the round number of the fit
    return {"server_round": int(server_round)}


def evaluate_config(server_round): # for the round number of the evaluate
    return {"server_round": int(server_round)}


def normalized(values): # normalizes the values to a probability distribution
    values = np.maximum(
        np.asarray(values, dtype=np.float64),
        0.0,
    )
    total = values.sum()
    if total <= 0:
        return np.ones(len(values), dtype=np.float64) / len(values)
    return values / total


def cosine_similarity(left, right): # computes the cosine similarity between two vectors
    denominator = np.linalg.norm(left) * np.linalg.norm(right)
    if denominator <= 0:
        return 0.0
    return float(np.dot(left, right) / denominator)


def mean_abs_shap(model, background, sample, number_of_features):
    """Return one mean absolute SHAP value per input feature."""
    values = shap.GradientExplainer(
        model,
        background,
    ).shap_values(sample)
    array = (
        np.stack(values, axis=0)
        if isinstance(values, list)
        else np.asarray(values)
    )

    if array.ndim == 3 and array.shape[1] == number_of_features:
        result = np.mean(np.abs(array), axis=(0, 2))
    elif array.ndim == 3 and array.shape[-1] == number_of_features:
        result = np.mean(np.abs(array), axis=(0, 1))
    elif array.ndim == 2 and array.shape[1] == number_of_features:
        result = np.mean(np.abs(array), axis=0)
    else:
        raise ValueError(f"Unexpected SHAP shape: {array.shape}")

    if result.shape != (number_of_features,):
        raise ValueError(
            f"SHAP produced {result.shape}; "
            f"expected ({number_of_features},)"
        )
    return result


def ranked_shap_rows(source, shap_values, server_round):
    """Print and return a ranked SHAP feature table."""
    shap_values = np.maximum(
        np.asarray(shap_values, dtype=np.float64),
        0.0,
    )
    importance_values = shap_values / max(shap_values.sum(), 1e-12)
    order = np.argsort(-shap_values, kind="stable")

    rows = []
    print(f"\n========== {source} | ROUND {server_round} SHAP ==========")
    for rank, feature_index in enumerate(order, start=1):
        feature = feature_names[feature_index]
        raw_value = float(shap_values[feature_index])
        importance = float(importance_values[feature_index])
        print(
            f"{rank:02d}. {feature}: "
            f"SHAP={raw_value:.8f}, "
            f"importance={importance * 100.0:.2f}%"
        )
        rows.append({
            "round": int(server_round),
            "source": source,
            "feature": feature,
            "mean_abs_shap": raw_value,
            "normalized_importance": importance,
            "importance_percent": importance * 100.0,
            "rank": int(rank),
        })
    return rows


# a balanced subset of the server validation data is used to compute the macro-F1 score for each client model, which is then used to weight the SHAP values in the aggregation step
utility_rng = np.random.RandomState(SEED)
utility_indices = []
for class_id in range(NUM_CLASSES):
    indices = np.where(y_val == class_id)[0]
    utility_indices.extend(
        utility_rng.choice(
            indices,
            size=min(2000, len(indices)),
            replace=False,
        ).tolist()
    )
utility_indices = np.asarray(utility_indices, dtype=np.int64)
x_utility = x_val[utility_indices]
y_utility = y_val[utility_indices]

# to score the validation macro-F1 for each client model, the server sets the model weights to the client's parameters and evaluates on the utility set
def validation_macro_f1(parameters):
    evaluation_model.set_weights(parameters)
    predictions = evaluation_model.predict(
        x_utility,
        batch_size=2048,
        verbose=0,
    ).argmax(axis=1)
    _, _, f1, _ = precision_recall_fscore_support(
        y_utility,
        predictions,
        average="macro",
        zero_division=0,
    )
    return float(f1)

# main evaluation function for the server, which evaluates the global model on the validation and test sets, computes metrics, saves results, and tracks the best validation F1 score
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
        RESULT_DIR / f"server_confusion_round_{server_round:03d}.csv",
    )

    if server_round > 0 and val_result["f1"] > best_val_f1:
        best_val_f1 = val_result["f1"]
        best_round = int(server_round)
        evaluation_model.save_weights(
            RESULT_DIR / "best_global.weights.h5"
        )

    print(
        f"[FedAvg Server] R{server_round}: "
        f"accuracy={test_result['accuracy']:.4f}, "
        f"balanced_accuracy={test_result['balanced_accuracy']:.4f}, "
        f"F1={test_result['f1']:.4f}"
    )

    # The parameters here are the newly aggregated round-10 global model.
    if int(server_round) == NUM_ROUNDS:
        server_shap_vector = mean_abs_shap(
            evaluation_model,
            server_shap_background,
            server_shap_sample,
            len(feature_names),
        )
        server_rows = ranked_shap_rows(
            source="Server global model (server test data)",
            shap_values=server_shap_vector,
            server_round=server_round,
        )
        pd.DataFrame(server_rows).to_csv(
            RESULT_DIR / "round10_server_global_shap_values.csv",
            index=False,
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


class ReliableShapFedAvg(fl.server.strategy.FedAvg):

    def aggregate_fit(self, server_round, results, failures):
        if not results:
            return None, {}

        records = []
        seen_ids = set()
        for _, fit_result in results:
            returned = dict(fit_result.metrics)
            client_id = int(returned["client_id"])
            if client_id in seen_ids:
                raise ValueError(f"Duplicate fit client ID: {client_id}")
            seen_ids.add(client_id)

            shap_vector = np.asarray([
                float(returned.get(f"shap::{name}", np.nan))
                for name in feature_names
            ], dtype=np.float64)
            if not np.all(np.isfinite(shap_vector)):
                raise ValueError(
                    f"Client {client_id} returned incomplete SHAP values."
                )
            records.append({
                "client_id": client_id,
                "num_examples": int(fit_result.num_examples),
                "parameters": parameters_to_ndarrays(
                    fit_result.parameters
                ),
                "shap": shap_vector,
            })
        records.sort(key=lambda record: record["client_id"])

        # These are the exact local-client vectors used for round-10 weighting.
        if int(server_round) == NUM_ROUNDS:
            client_rows = []
            for record in records:
                client_rows.extend(
                    ranked_shap_rows(
                        source=f"Client {record['client_id']} local model",
                        shap_values=record["shap"],
                        server_round=server_round,
                    )
                )
            pd.DataFrame(client_rows).to_csv(
                RESULT_DIR / "round10_client_shap_values.csv",
                index=False,
            )

        size_weights = normalized([
            record["num_examples"] for record in records
        ])

        shap_matrix = np.stack([
            record["shap"] for record in records
        ])
        shap_matrix /= np.maximum(
            np.linalg.norm(shap_matrix, axis=1, keepdims=True),
            1e-12,
        )
        consensus = np.median(shap_matrix, axis=0)
        alignments = np.asarray([
            max(0.0, cosine_similarity(vector, consensus))
            for vector in shap_matrix
        ])

        validation_f1 = np.asarray([
            validation_macro_f1(record["parameters"])
            for record in records
        ])
        reliable_shap_weights = normalized(
            alignments * np.maximum(validation_f1, 1e-6)
        )

        # SHAP is chosen by the blend coefficient; FedAvg remains the anchor.
        final_weights = normalized(
    (1.0 - SHAP_BLEND) * size_weights
    + SHAP_BLEND * reliable_shap_weights
)

        aggregated = [
            np.zeros_like(layer)
            for layer in records[0]["parameters"]
        ]
        for record_index, (record, client_weight) in enumerate(
            zip(records, final_weights)
        ):
            for layer_index, layer in enumerate(record["parameters"]):
                aggregated[layer_index] += layer * client_weight
            aggregation_history.append({
                "round": int(server_round),
                "client_id": record["client_id"],
                "num_examples": record["num_examples"],
                "fedavg_size_weight": float(
                    size_weights[record_index]
                ),
                "shap_alignment": float(
                    alignments[record_index]
                ),
                "server_validation_f1": float(
                    validation_f1[record_index]
                ),
                "reliable_shap_weight": float(
                    reliable_shap_weights[record_index]
                ),
                "final_weight": float(
                    final_weights[record_index]
                ),
            })

        pd.DataFrame(aggregation_history).to_csv(
            RESULT_DIR / "aggregation_weights.csv",
            index=False,
        )
        print(
            f"[Reliable SHAP] R{server_round}: "
            + ", ".join(
                f"C{record['client_id']}={weight:.4f}"
                for record, weight in zip(records, final_weights)
            )
        )
        return ndarrays_to_parameters(aggregated), {}

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
                    f"Client {client_id}: invalid confusion matrix"
                )
            save_matrix(
                matrix,
                RESULT_DIR
                / f"client_{client_id}_confusion_round_{server_round:03d}.csv",
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


strategy = ReliableShapFedAvg(
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
    f"[FedAvg Server] Best validation macro-F1="
    f"{best_val_f1:.4f} at round {best_round}"
)


