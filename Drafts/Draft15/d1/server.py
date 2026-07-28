"""Server-side evaluation, SHAP aggregation, and final SHAP comparison."""
import json
import sys
from pathlib import Path

import flwr as fl
import joblib
import numpy as np
import pandas as pd
import shap
import tensorflow as tf
from flwr.common import ndarrays_to_parameters, parameters_to_ndarrays
from sklearn.metrics import (
    accuracy_score, balanced_accuracy_score, confusion_matrix,
    precision_recall_fscore_support,
)
from sklearn.model_selection import train_test_split

from Labels7 import ALL_LABELS, NUM_CLASSES, NUM_FEATURES, le
from model import build_model

NUM_CLIENTS = 5
NUM_ROUNDS = 10
PROXIMAL_MU = 0.01
SIZE_INFLUENCE = 0.50
SHAP_INFLUENCE = 0.50
SEED = 42
SERVER_DATA_PATH = r"D:\CAPSTONE\Server_Test_7Class.csv"
SCALER_PATH = r"D:\CAPSTONE\global_scaler_7Class.pkl"
RESULT_DIR = Path("results_10round_7class")
RESULT_DIR.mkdir(parents=True, exist_ok=True)

np.random.seed(SEED)
tf.keras.utils.set_random_seed(SEED)


def metrics(y_true, y_pred):
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
        "confusion_matrix": matrix,
    }


def mean_abs_shap(model, background, sample, number_of_features):
    values = shap.GradientExplainer(model, background).shap_values(sample)
    array = np.stack(values, axis=0) if isinstance(values, list) else np.asarray(values)
    if array.ndim == 3 and array.shape[-1] == number_of_features:
        result = np.mean(np.abs(array), axis=(0, 1))
    elif array.ndim == 3 and array.shape[1] == number_of_features:
        result = np.mean(np.abs(array), axis=(0, 2))
    elif array.ndim == 2 and array.shape[1] == number_of_features:
        result = np.mean(np.abs(array), axis=0)
    else:
        raise ValueError(f"Unexpected SHAP output shape: {array.shape}")
    return np.asarray(result, dtype=np.float64)


def normalized(values):
    values = np.asarray(values, dtype=np.float64)
    values = np.maximum(values, 0.0)
    total = values.sum()
    return values / total if total > 0 else np.ones(len(values)) / len(values)


def cosine_similarity(left, right):
    denominator = np.linalg.norm(left) * np.linalg.norm(right)
    return float(np.dot(left, right) / denominator) if denominator else 0.0


def save_matrix(matrix, path):
    pd.DataFrame(
        matrix, index=[f"true_{name}" for name in ALL_LABELS],
        columns=[f"pred_{name}" for name in ALL_LABELS],
    ).to_csv(path)


# The split script created this server-only holdout before Dirichlet partitioning.
server_df = pd.read_csv(SERVER_DATA_PATH).replace(
    [np.inf, -np.inf], np.nan
).dropna()
server_df.drop(
    columns=[c for c in server_df if c.lower().startswith("unnamed:")],
    inplace=True, errors="ignore",
)
feature_names = [c for c in server_df.columns if c != "Label"]
if len(feature_names) != NUM_FEATURES:
    raise ValueError(f"Expected {NUM_FEATURES} server features, found {len(feature_names)}")

server_labels = server_df["Label"].astype(str).str.strip().str.upper()
unknown = sorted(set(server_labels) - set(ALL_LABELS))
if unknown:
    raise ValueError(f"Unexpected server family labels: {unknown}")

scaler = joblib.load(SCALER_PATH)
server_x = scaler.transform(server_df[feature_names]).astype(np.float32)
server_y = le.transform(server_labels).astype(np.int32)
x_val, x_test, y_val, y_test = train_test_split(
    server_x, server_y, test_size=0.5, random_state=SEED, stratify=server_y
)

eval_model = build_model(NUM_FEATURES, NUM_CLASSES)
eval_model.compile(loss="sparse_categorical_crossentropy")
initial_parameters = ndarrays_to_parameters(eval_model.get_weights())

rng = np.random.RandomState(SEED)
background_idx = rng.choice(len(x_val), min(50, len(x_val)), replace=False)
sample_idx = rng.choice(len(x_test), min(100, len(x_test)), replace=False)
server_shap_background = x_val[background_idx]
server_shap_sample = x_test[sample_idx]

server_history = []
client_history = []
aggregation_history = []
class_aggregation_history = []
class_shap_history = []
latest_client_shap = {}
best_val_f1 = -1.0
best_round = 0
best_weights = None
last_global_weights = None


def fit_config(server_round):
    return {"server_round": int(server_round), "proximal_mu": float(PROXIMAL_MU)}


def evaluate_config(server_round):
    return {"server_round": int(server_round)}


def centralized_evaluate(server_round, parameters, config):
    global best_val_f1, best_round, best_weights, last_global_weights
    eval_model.set_weights(parameters)
    if server_round > 0:
        last_global_weights = [weight.copy() for weight in parameters]

    val_probabilities = eval_model.predict(x_val, batch_size=2048, verbose=0)
    test_probabilities = eval_model.predict(x_test, batch_size=2048, verbose=0)
    val_result = metrics(y_val, val_probabilities.argmax(axis=1))
    test_result = metrics(y_test, test_probabilities.argmax(axis=1))
    test_loss = float(np.mean(
        tf.keras.losses.sparse_categorical_crossentropy(y_test, test_probabilities)
    ))

    row = {"round": int(server_round)}
    for name in ("accuracy", "balanced_accuracy", "precision", "recall", "f1"):
        row[f"val_{name}"] = val_result[name]
        row[f"test_{name}"] = test_result[name]
    row["test_loss"] = test_loss
    server_history.append(row)
    pd.DataFrame(server_history).to_csv(RESULT_DIR / "server_metrics.csv", index=False)
    save_matrix(
        test_result["confusion_matrix"],
        RESULT_DIR / f"server_confusion_round_{server_round:03d}.csv",
    )

    if server_round > 0 and val_result["f1"] > best_val_f1:
        best_val_f1 = val_result["f1"]
        best_round = int(server_round)
        best_weights = [weight.copy() for weight in parameters]
        eval_model.save_weights(RESULT_DIR / "best_global.weights.h5")

    print(
        f"[Server] R{server_round}: test accuracy={test_result['accuracy']:.4f}, "
        f"precision={test_result['precision']:.4f}, recall={test_result['recall']:.4f}, "
        f"F1={test_result['f1']:.4f}"
    )
    return test_loss, {
        key: test_result[key]
        for key in ("accuracy", "balanced_accuracy", "precision", "recall", "f1")
    }


class ShapWeightedStrategy(fl.server.strategy.FedAvg):
    def aggregate_fit(self, server_round, results, failures):
        if not results:
            return None, {}

        records = []
        seen_ids = set()
        for _, fit_result in results:
            fit_metrics = dict(fit_result.metrics)
            if "client_id" not in fit_metrics:
                raise ValueError("A client did not return client_id")
            client_id = int(fit_metrics["client_id"])
            if client_id not in range(1, NUM_CLIENTS + 1) or client_id in seen_ids:
                raise ValueError(f"Invalid or duplicate client_id: {client_id}")
            seen_ids.add(client_id)

            shap_vector = np.asarray([
                float(fit_metrics.get(f"shap::{name}", np.nan))
                for name in feature_names
            ])
            if not np.all(np.isfinite(shap_vector)):
                raise ValueError(f"Client {client_id} returned incomplete SHAP values")

            missing_count_keys = [
                f"class_count::{class_name}"
                for class_name in ALL_LABELS
                if f"class_count::{class_name}" not in fit_metrics
            ]
            if missing_count_keys:
                raise ValueError(
                    f"Client {client_id} did not return class counts: "
                    f"{missing_count_keys}"
                )

            records.append({
                "client_id": client_id,
                "num_examples": int(fit_result.num_examples),
                "parameters": parameters_to_ndarrays(fit_result.parameters),
                "shap": shap_vector,
                "class_counts": np.asarray([
                    int(fit_metrics.get(f"class_count::{class_name}", 0))
                    for class_name in ALL_LABELS
                ], dtype=np.float64),
                "class_shap": {
                    class_index: np.asarray([
                        float(
                            fit_metrics[
                                f"class_shap::{class_name}::{feature_name}"
                            ]
                        )
                        for feature_name in feature_names
                    ], dtype=np.float64)
                    for class_index, class_name in enumerate(ALL_LABELS)
                    if int(
                        fit_metrics.get(f"class_count::{class_name}", 0)
                    ) > 0
                },
                "local_accuracy": float(fit_metrics.get("local_accuracy", 0.0)),
                "local_precision": float(fit_metrics.get("local_precision", 0.0)),
                "local_recall": float(fit_metrics.get("local_recall", 0.0)),
                "local_f1": float(fit_metrics.get("local_f1", 0.0)),
            })
        records.sort(key=lambda item: item["client_id"])

        shap_matrix = np.stack([record["shap"] for record in records])
        shap_normalized = shap_matrix / np.maximum(
            np.linalg.norm(shap_matrix, axis=1, keepdims=True), 1e-12
        )
        # Median consensus is robust to one unusual or malicious client.
        consensus = np.median(shap_normalized, axis=0)
        alignments = np.asarray([
            max(0.0, cosine_similarity(vector, consensus))
            for vector in shap_normalized
        ])
        # Square-root weighting keeps data volume relevant without letting
        # the two largest non-IID clients dominate the global model.
        size_weights = normalized([
            np.sqrt(record["num_examples"])
            for record in records
        ])
        shap_weights = normalized(alignments)
        final_weights = normalized(
            SIZE_INFLUENCE * size_weights + SHAP_INFLUENCE * shap_weights
        )

        aggregated = [
            np.zeros_like(layer)
            for layer in records[0]["parameters"]
        ]

        # Shared representation: aggregate with the ordinary blended
        # square-root-size and SHAP-consensus client weights.
        for record, client_weight in zip(records, final_weights):
            for layer_index in range(len(aggregated) - 2):
                layer = record["parameters"][layer_index]
                aggregated[layer_index] += layer * client_weight

            latest_client_shap[record["client_id"]] = record["shap"].copy()
            aggregation_history.append({
                "round": int(server_round),
                "client_id": record["client_id"],
                "num_examples": record["num_examples"],
                "size_weight": float(size_weights[record["client_id"] - 1]),
                "shap_consensus_alignment": float(
                    alignments[record["client_id"] - 1]
                ),
                "shap_weight": float(shap_weights[record["client_id"] - 1]),
                "final_weight": float(final_weights[record["client_id"] - 1]),
                "local_accuracy": record["local_accuracy"],
                "local_precision": record["local_precision"],
                "local_recall": record["local_recall"],
                "local_f1": record["local_f1"],
            })

        # Classification head: aggregate each output class separately.
        # The last two arrays are the output Dense kernel (hidden x classes)
        # and bias (classes). Clients without examples of a class receive zero
        # influence over that class's output parameters.
        output_kernel = np.zeros_like(records[0]["parameters"][-2])
        output_bias = np.zeros_like(records[0]["parameters"][-1])

        for class_index, class_name in enumerate(ALL_LABELS):
            counts = np.asarray([
                record["class_counts"][class_index]
                for record in records
            ], dtype=np.float64)
            present_mask = (counts > 0).astype(np.float64)
            class_shap_weights = np.zeros(len(records), dtype=np.float64)

            if present_mask.sum() == 0:
                class_weights = final_weights.copy()
            else:
                class_size_weights = normalized(
                    np.sqrt(counts) * present_mask
                )

                # Compare only explanations for this output class, and only
                # across clients that actually contain this class.
                present_indices = np.where(present_mask > 0)[0]
                class_vectors = np.stack([
                    records[index]["class_shap"][class_index]
                    for index in present_indices
                ])
                class_vectors = class_vectors / np.maximum(
                    np.linalg.norm(
                        class_vectors,
                        axis=1,
                        keepdims=True,
                    ),
                    1e-12,
                )
                class_consensus = np.median(class_vectors, axis=0)
                class_alignments = np.asarray([
                    max(0.0, cosine_similarity(vector, class_consensus))
                    for vector in class_vectors
                ])
                class_shap_weights[present_indices] = normalized(
                    class_alignments
                )
                class_weights = normalized(
                    0.50 * class_size_weights
                    + 0.50 * class_shap_weights
                )

            for record_index, (record, class_weight) in enumerate(
                zip(records, class_weights)
            ):
                output_kernel[:, class_index] += (
                    record["parameters"][-2][:, class_index]
                    * class_weight
                )
                output_bias[class_index] += (
                    record["parameters"][-1][class_index]
                    * class_weight
                )
                class_aggregation_history.append({
                    "round": int(server_round),
                    "class": class_name,
                    "client_id": record["client_id"],
                    "class_examples": int(
                        record["class_counts"][class_index]
                    ),
                    "class_weight": float(class_weight),
                    "class_shap_weight": float(
                        class_shap_weights[record_index]
                    ),
                })

                if record["class_counts"][class_index] > 0:
                    for feature_index, feature_name in enumerate(feature_names):
                        class_shap_history.append({
                            "round": int(server_round),
                            "client_id": record["client_id"],
                            "class": class_name,
                            "feature": feature_name,
                            "class_shap_value": float(
                                record["class_shap"][class_index][feature_index]
                            ),
                        })

        aggregated[-2] = output_kernel
        aggregated[-1] = output_bias

        pd.DataFrame(aggregation_history).to_csv(
            RESULT_DIR / "aggregation_weights.csv", index=False
        )
        pd.DataFrame(class_aggregation_history).to_csv(
            RESULT_DIR / "class_aggregation_weights.csv", index=False
        )
        pd.DataFrame(class_shap_history).to_csv(
            RESULT_DIR / "class_specific_shap_values.csv", index=False
        )
        print(
            f"[Server] R{server_round} SHAP weights: "
            + ", ".join(
                f"C{record['client_id']}={weight:.4f}"
                for record, weight in zip(records, final_weights)
            )
        )
        return ndarrays_to_parameters(aggregated), {}

    def aggregate_evaluate(self, server_round, results, failures):
        if not results:
            return None, {}

        round_ids = set()
        for _, evaluate_result in results:
            returned = dict(evaluate_result.metrics)
            if "client_id" not in returned:
                raise ValueError("A client evaluation did not return client_id")
            client_id = int(returned["client_id"])
            if client_id in round_ids:
                raise ValueError(f"Duplicate evaluation client_id: {client_id}")
            round_ids.add(client_id)

            matrix = np.asarray(json.loads(returned["confusion_matrix"]), dtype=int)
            if matrix.shape != (NUM_CLASSES, NUM_CLASSES):
                raise ValueError(f"Client {client_id}: invalid confusion matrix shape")
            save_matrix(
                matrix,
                RESULT_DIR / f"client_{client_id}_confusion_round_{server_round:03d}.csv",
            )
            row = {
                "round": int(server_round),
                "client_id": client_id,
                "num_examples": int(evaluate_result.num_examples),
                "loss": float(evaluate_result.loss),
                **{
                    name: float(returned.get(name, 0.0))
                    for name in (
                        "accuracy", "balanced_accuracy",
                        "precision", "recall", "f1",
                    )
                },
            }

            personalized_matrix_json = returned.get(
                "personalized_confusion_matrix"
            )
            if personalized_matrix_json is not None:
                personalized_matrix = np.asarray(
                    json.loads(personalized_matrix_json),
                    dtype=int,
                )
                if personalized_matrix.shape != (NUM_CLASSES, NUM_CLASSES):
                    raise ValueError(
                        f"Client {client_id}: invalid personalized "
                        "confusion matrix shape"
                    )
                save_matrix(
                    personalized_matrix,
                    RESULT_DIR
                    / f"client_{client_id}_personalized_confusion_final.csv",
                )
                row.update({
                    "personalized_loss": float(
                        returned.get("personalized_loss", 0.0)
                    ),
                    **{
                        f"personalized_{name}": float(
                            returned.get(f"personalized_{name}", 0.0)
                        )
                        for name in (
                            "accuracy", "balanced_accuracy",
                            "precision", "recall", "f1",
                        )
                    },
                })

            client_history.append(row)
        client_history.sort(key=lambda row: (row["round"], row["client_id"]))
        pd.DataFrame(client_history).to_csv(
            RESULT_DIR / "client_metrics.csv", index=False
        )
        return super().aggregate_evaluate(server_round, results, failures)


def final_shap_comparison():
    if last_global_weights is None or len(latest_client_shap) != NUM_CLIENTS:
        print("[Server] Final SHAP comparison skipped: required values are missing.")
        return

    # Compare the final aggregated global model with final-round client models.
    eval_model.set_weights(last_global_weights)
    server_vector = mean_abs_shap(
        eval_model, server_shap_background, server_shap_sample, len(feature_names)
    )
    server_rank = pd.Series(server_vector).rank(method="average")
    top_count = min(10, len(feature_names))
    server_top = set(np.argsort(server_vector)[-top_count:])

    summary_rows = []
    feature_rows = []
    for client_id in sorted(latest_client_shap):
        client_vector = latest_client_shap[client_id]
        client_rank = pd.Series(client_vector).rank(method="average")
        spearman = float(server_rank.corr(client_rank, method="pearson"))
        client_top = set(np.argsort(client_vector)[-top_count:])
        top_overlap = len(server_top & client_top) / top_count
        summary_rows.append({
            "client_id": client_id,
            "server_round": NUM_ROUNDS,
            "cosine_similarity": cosine_similarity(server_vector, client_vector),
            "spearman_rank_correlation": spearman,
            f"top_{top_count}_overlap": top_overlap,
        })
        for index, feature in enumerate(feature_names):
            feature_rows.append({
                "client_id": client_id,
                "feature": feature,
                "server_shap": float(server_vector[index]),
                "client_shap": float(client_vector[index]),
                "absolute_difference": float(
                    abs(server_vector[index] - client_vector[index])
                ),
            })

    pd.DataFrame(summary_rows).to_csv(
        RESULT_DIR / "final_server_client_shap_comparison.csv", index=False
    )
    pd.DataFrame(feature_rows).to_csv(
        RESULT_DIR / "final_server_client_shap_by_feature.csv", index=False
    )
    pd.DataFrame({
        "feature": feature_names, "server_shap": server_vector
    }).sort_values("server_shap", ascending=False).to_csv(
        RESULT_DIR / "final_server_shap.csv", index=False
    )
    print("\nFinal server/client SHAP comparison:")
    print(pd.DataFrame(summary_rows).round(4).to_string(index=False))


strategy = ShapWeightedStrategy(
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

final_shap_comparison()
print(f"[Server] Best validation macro-F1={best_val_f1:.4f} at round {best_round}")
