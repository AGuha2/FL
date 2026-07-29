"""Seven-class FL server with class-specific reliable SHAP aggregation."""
import json
import sys
from pathlib import Path

import flwr as fl
import joblib
import numpy as np
import pandas as pd
import tensorflow as tf
from flwr.common import ndarrays_to_parameters, parameters_to_ndarrays
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    confusion_matrix,
    precision_recall_fscore_support,
)
from sklearn.model_selection import train_test_split

from Labels7 import ALL_LABELS, NUM_CLASSES, NUM_FEATURES, le
from class_specific_shap_weighting import (
    calculate_client_weights,
    class_f1,
    class_specific_mean_abs_shap,
)
from model import build_model
from posthoc_shap_analysis import save_final_posthoc_shap_analysis


NUM_CLIENTS = 5
NUM_ROUNDS = 10
SEED = 42

SERVER_DATA_PATH = r"D:\CAPSTONE\Server_Test_7Class.csv"
SCALER_PATH = r"D:\CAPSTONE\global_scaler_7Class.pkl"
RESULT_DIR = Path(
    "results_10round_7class_equal_reliable_consensus_shap"
)
RESULT_DIR.mkdir(parents=True, exist_ok=True)

# Aggregation starts in round 1. Before round 1, every class has the same
# minimum server F1, so no class is initially preferred.
WARMUP_ROUNDS = 0
INITIAL_SERVER_CLASS_F1 = 0.05
SHAP_BLEND = 0.50
SHAP_TEMPERATURE = 0.25
MIN_CLIENT_WEIGHT = 0.12
MAX_CLIENT_WEIGHT = 0.28
WEIGHT_SMOOTHING = 0.70

# Each client update is evaluated on the same balanced validation subset.
UTILITY_SAMPLES_PER_CLASS = 2000

# The separate final test partition is used only after the last aggregation.
POSTHOC_SHAP_SAMPLES_PER_CLASS = 50
POSTHOC_SHAP_BACKGROUND_SIZE = 50

np.random.seed(SEED)
tf.keras.utils.set_random_seed(SEED)


def calculate_metrics(y_true, y_pred):
    precision_by_class, recall_by_class, f1_by_class, _ = (
        precision_recall_fscore_support(
            y_true,
            y_pred,
            labels=np.arange(NUM_CLASSES),
            average=None,
            zero_division=0,
        )
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
        "precision": float(np.mean(precision_by_class)),
        "recall": float(np.mean(recall_by_class)),
        "f1": float(np.mean(f1_by_class)),
        "class_f1": np.asarray(f1_by_class, dtype=np.float64),
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
                f"Server data has no {ALL_LABELS[class_id]}."
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

# Common balanced validation subset for evaluating every submitted client
# model. The final server test partition is never used for this calculation.
utility_indices = balanced_indices(
    y_val,
    UTILITY_SAMPLES_PER_CLASS,
    SEED + 1000,
)
x_utility = x_val[utility_indices]
y_utility = y_val[utility_indices]

# A larger, final-round-only SHAP sample is drawn from the untouched test
# partition. It is not used for aggregation, model selection, or training.
posthoc_server_indices = balanced_indices(
    y_test,
    POSTHOC_SHAP_SAMPLES_PER_CLASS,
    SEED + 4000,
)
posthoc_server_samples = x_test[posthoc_server_indices]
posthoc_server_labels = y_test[posthoc_server_indices]

posthoc_background_rng = np.random.RandomState(SEED + 5000)
posthoc_background_pool = np.setdiff1d(
    np.arange(len(x_test)),
    posthoc_server_indices,
)
if len(posthoc_background_pool) == 0:
    posthoc_background_pool = np.arange(len(x_test))
posthoc_server_background_indices = posthoc_background_rng.choice(
    posthoc_background_pool,
    size=min(
        POSTHOC_SHAP_BACKGROUND_SIZE,
        len(posthoc_background_pool),
    ),
    replace=False,
)
posthoc_server_background = x_test[
    posthoc_server_background_indices
]


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
server_f1_state = {
    "round": 0,
    "values": np.full(
        NUM_CLASSES,
        INITIAL_SERVER_CLASS_F1,
        dtype=np.float64,
    ),
}


def fit_config(server_round):
    return {
        "server_round": int(server_round),
        "run_posthoc_shap": bool(server_round == NUM_ROUNDS),
    }


def evaluate_config(server_round):
    return {"server_round": int(server_round)}


def centralized_evaluate(server_round, parameters, config):
    global best_val_f1, best_round

    # Flower can evaluate the random initial model before round 1. It is
    # intentionally skipped so round 1 uses INITIAL_SERVER_CLASS_F1.
    if server_round == 0:
        return None

    evaluation_model.set_weights(parameters)

    val_probabilities = evaluation_model.predict(
        x_val,
        batch_size=2048,
        verbose=0,
    )
    val_result = calculate_metrics(
        y_val,
        val_probabilities.argmax(axis=1),
    )
    val_loss = float(np.mean(
        tf.keras.losses.sparse_categorical_crossentropy(
            y_val,
            val_probabilities,
        )
    ))

    row = {
        "round": int(server_round),
        "val_loss": val_loss,
    }
    for name in (
        "accuracy",
        "balanced_accuracy",
        "precision",
        "recall",
        "f1",
    ):
        row[f"val_{name}"] = val_result[name]
    for class_id, class_name in enumerate(ALL_LABELS):
        row[f"val_f1::{class_name}"] = float(
            val_result["class_f1"][class_id]
        )

    save_matrix(
        val_result["confusion_matrix"],
        RESULT_DIR
        / f"server_validation_confusion_round_{server_round:03d}.csv",
    )

    # These values are read by aggregate_fit in the NEXT round.
    server_f1_state["round"] = int(server_round)
    server_f1_state["values"] = val_result["class_f1"].copy()

    if val_result["f1"] > best_val_f1:
        best_val_f1 = val_result["f1"]
        best_round = int(server_round)
        evaluation_model.save_weights(
            RESULT_DIR / "best_global.weights.h5"
        )

    # The final test partition is touched only after round 10 aggregation.
    if server_round == NUM_ROUNDS:
        test_probabilities = evaluation_model.predict(
            x_test,
            batch_size=2048,
            verbose=0,
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
        row["test_loss"] = test_loss
        for name in (
            "accuracy",
            "balanced_accuracy",
            "precision",
            "recall",
            "f1",
        ):
            row[f"test_{name}"] = test_result[name]
        for class_id, class_name in enumerate(ALL_LABELS):
            row[f"test_f1::{class_name}"] = float(
                test_result["class_f1"][class_id]
            )
        save_matrix(
            test_result["confusion_matrix"],
            RESULT_DIR / "final_server_test_confusion.csv",
        )
        print(
            "[Consensus SHAP Server] Final test: "
            f"accuracy={test_result['accuracy']:.4f}, "
            f"macro_F1={test_result['f1']:.4f}"
        )

    server_history.append(row)
    pd.DataFrame(server_history).to_csv(
        RESULT_DIR / "server_metrics.csv",
        index=False,
    )

    print(
        f"[Consensus SHAP Server] R{server_round}: "
        f"validation_accuracy={val_result['accuracy']:.4f}, "
        f"validation_F1={val_result['f1']:.4f}"
    )
    return val_loss, {
        name: val_result[name]
        for name in (
            "accuracy",
            "balanced_accuracy",
            "precision",
            "recall",
            "f1",
        )
    }


class ClassSpecificReliableShap(fl.server.strategy.FedAvg):
    """FedAvg blended with validation reliability and consensus SHAP."""

    def __init__(
        self,
        *,
        evaluation_model,
        server_f1_state,
        x_utility,
        y_utility,
        posthoc_shap_background,
        posthoc_shap_samples,
        posthoc_shap_labels,
        feature_names,
        class_names,
        result_dir,
        final_round=10,
        warmup_rounds=0,
        shap_blend=0.50,
        temperature=0.25,
        minimum_weight=0.12,
        maximum_weight=0.28,
        smoothing=0.70,
        **fedavg_arguments,
    ):
        super().__init__(**fedavg_arguments)
        self.model = evaluation_model
        self.server_f1_state = server_f1_state
        self.x_utility = x_utility
        self.y_utility = y_utility
        self.posthoc_shap_background = posthoc_shap_background
        self.posthoc_shap_samples = posthoc_shap_samples
        self.posthoc_shap_labels = posthoc_shap_labels
        self.feature_names = list(feature_names)
        self.class_names = list(class_names)
        self.num_classes = len(self.class_names)
        self.result_dir = Path(result_dir)
        self.final_round = int(final_round)
        self.warmup_rounds = int(warmup_rounds)
        self.shap_blend = float(shap_blend)
        self.temperature = float(temperature)
        self.minimum_weight = float(minimum_weight)
        self.maximum_weight = float(maximum_weight)
        self.smoothing = float(smoothing)
        self.previous_weights = None
        self.global_parameters = parameters_to_ndarrays(
            fedavg_arguments["initial_parameters"]
        )
        self.aggregation_history = []
        self.posthoc_complete = False

    def _class_f1(self, parameters):
        self.model.set_weights(parameters)
        probabilities = self.model.predict(
            self.x_utility,
            batch_size=2048,
            verbose=0,
        )
        return class_f1(
            self.y_utility,
            probabilities.argmax(axis=1),
            self.num_classes,
        )

    def _posthoc_server_shap(self):
        self.model.set_weights(self.global_parameters)
        matrix, _ = class_specific_mean_abs_shap(
            self.model,
            self.posthoc_shap_background,
            self.posthoc_shap_samples,
            self.posthoc_shap_labels,
            self.num_classes,
            len(self.feature_names),
        )
        return matrix

    def _read_client_shap(self, metrics, namespace="shap"):
        matrix = np.zeros(
            (self.num_classes, len(self.feature_names)),
            dtype=np.float64,
        )
        available = np.zeros(self.num_classes, dtype=np.float64)
        for class_id, class_name in enumerate(self.class_names):
            available[class_id] = float(
                metrics.get(
                    f"{namespace}_available::{class_name}",
                    0.0,
                )
            )
            for feature_id, feature_name in enumerate(
                self.feature_names
            ):
                key = f"{namespace}::{class_name}::{feature_name}"
                matrix[class_id, feature_id] = float(
                    metrics.get(key, 0.0)
                )
        if not np.all(np.isfinite(matrix)):
            raise ValueError(
                "A client returned non-finite SHAP values."
            )
        return matrix, available

    def aggregate_fit(self, server_round, results, failures):
        if not results:
            return None, {}

        records = []
        seen = set()
        for _, fit_result in results:
            metrics = dict(fit_result.metrics)
            client_id = int(metrics["client_id"])
            if client_id in seen:
                raise ValueError(f"Duplicate client ID {client_id}")
            seen.add(client_id)

            shap_matrix, available = self._read_client_shap(metrics)
            posthoc_shap = shap_matrix
            posthoc_available = available
            if (
                server_round == self.final_round
                and any(
                    key.startswith("posthoc_shap::")
                    for key in metrics
                )
            ):
                posthoc_shap, posthoc_available = (
                    self._read_client_shap(
                        metrics,
                        namespace="posthoc_shap",
                    )
                )

            records.append({
                "client_id": client_id,
                "num_examples": int(fit_result.num_examples),
                "parameters": parameters_to_ndarrays(
                    fit_result.parameters
                ),
                "shap": shap_matrix,
                "available": available,
                "posthoc_shap": posthoc_shap,
                "posthoc_available": posthoc_available,
            })

        records.sort(key=lambda item: item["client_id"])

        size_weights = np.asarray(
            [item["num_examples"] for item in records],
            dtype=np.float64,
        )
        size_weights /= size_weights.sum()

        if server_round <= self.warmup_rounds:
            final_weights = size_weights
            diagnostic = None
        else:
            # All submitted client models are evaluated on one common,
            # balanced server-validation subset. This makes their per-class
            # F1 scores directly comparable without receiving client data.
            client_class_f1 = np.stack([
                self._class_f1(item["parameters"])
                for item in records
            ])
            final_weights, diagnostic = calculate_client_weights(
                size_weights=size_weights,
                client_shap=np.stack([
                    item["shap"] for item in records
                ]),
                available=np.stack([
                    item["available"] for item in records
                ]),
                client_class_f1=client_class_f1,
                previous_server_class_f1=np.asarray(
                    self.server_f1_state["values"],
                    dtype=np.float64,
                ),
                previous_weights=self.previous_weights,
                shap_blend=self.shap_blend,
                temperature=self.temperature,
                minimum=self.minimum_weight,
                maximum=self.maximum_weight,
                smoothing=self.smoothing,
            )

        self.previous_weights = final_weights.copy()
        aggregated = [
            np.zeros_like(layer)
            for layer in records[0]["parameters"]
        ]
        for item, weight in zip(records, final_weights):
            for layer_id, layer in enumerate(item["parameters"]):
                aggregated[layer_id] += layer * weight

        self.global_parameters = aggregated

        # Final post-hoc analysis runs only after the last global model
        # has been aggregated. It cannot influence aggregation weights.
        if (
            server_round == self.final_round
            and not self.posthoc_complete
        ):
            self.model.set_weights(self.global_parameters)
            final_server_shap = self._posthoc_server_shap()
            final_client_records = [
                {
                    "client_id": item["client_id"],
                    "shap": item["posthoc_shap"],
                    "available": item["posthoc_available"],
                }
                for item in records
            ]
            output_paths = save_final_posthoc_shap_analysis(
                self.result_dir,
                server_round=server_round,
                feature_names=self.feature_names,
                class_names=self.class_names,
                server_shap=final_server_shap,
                client_records=final_client_records,
            )
            self.model.set_weights(self.global_parameters)
            self.model.save_weights(
                self.result_dir / "final_global.weights.h5"
            )
            self.posthoc_complete = True
            print(
                "[Final post-hoc SHAP] Saved server/client analysis: "
                + ", ".join(
                    path.name for path in output_paths.values()
                )
            )

        for index, item in enumerate(records):
            row = {
                "round": int(server_round),
                "client_id": item["client_id"],
                "num_examples": item["num_examples"],
                "fedavg_weight": float(size_weights[index]),
                "final_weight": float(final_weights[index]),
                "warmup": int(
                    server_round <= self.warmup_rounds
                ),
                "server_f1_source_round": int(
                    self.server_f1_state["round"]
                ),
            }
            if diagnostic is not None:
                row["client_score"] = float(
                    diagnostic["score"][index]
                )
                row["raw_shap_weight"] = float(
                    diagnostic["shap_weight"][index]
                )
                for class_id, class_name in enumerate(
                    self.class_names
                ):
                    row[f"importance::{class_name}"] = float(
                        diagnostic["importance"][class_id]
                    )
                    row[f"previous_server_f1::{class_name}"] = float(
                        diagnostic[
                            "previous_server_class_f1"
                        ][class_id]
                    )
                    row[f"similarity::{class_name}"] = float(
                        diagnostic["similarity"][index, class_id]
                    )
                    row[f"client_validation_f1::{class_name}"] = float(
                        diagnostic["reliability"][index, class_id]
                    )
                    row[f"gain::{class_name}"] = float(
                        diagnostic["gain"][index, class_id]
                    )
                    row[f"peer_count::{class_name}"] = int(
                        diagnostic["peer_count"][index, class_id]
                    )
                    row[f"comparable::{class_name}"] = int(
                        diagnostic["comparable"][index, class_id]
                    )
            self.aggregation_history.append(row)

        pd.DataFrame(self.aggregation_history).to_csv(
            self.result_dir / "aggregation_weights.csv",
            index=False,
        )
        print(
            f"[Reliable consensus SHAP] R{server_round}: "
            + ", ".join(
                f"C{item['client_id']}={weight:.4f}"
                for item, weight in zip(records, final_weights)
            )
        )
        return ndarrays_to_parameters(aggregated), {}


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
    server_f1_state=server_f1_state,
    x_utility=x_utility,
    y_utility=y_utility,
    posthoc_shap_background=posthoc_server_background,
    posthoc_shap_samples=posthoc_server_samples,
    posthoc_shap_labels=posthoc_server_labels,
    feature_names=feature_names,
    class_names=ALL_LABELS,
    result_dir=RESULT_DIR,
    final_round=NUM_ROUNDS,
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
