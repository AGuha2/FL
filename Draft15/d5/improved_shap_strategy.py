"""Flower strategy for reliability-gated class-specific SHAP aggregation."""
from __future__ import annotations

import json
from pathlib import Path

import flwr as fl
import numpy as np
import pandas as pd
from flwr.common import ndarrays_to_parameters, parameters_to_ndarrays

from class_specific_shap_weighting import (
    calculate_client_weights,
    class_f1,
    class_specific_mean_abs_shap,
)


class ClassSpecificReliableShap(fl.server.strategy.FedAvg):
    def __init__(
        self,
        *,
        evaluation_model,
        x_utility,
        y_utility,
        shap_background,
        shap_samples,
        shap_labels,
        feature_names,
        class_names,
        result_dir,
        warmup_rounds=2,
        shap_blend=0.50,
        temperature=0.25,
        minimum_weight=0.12,
        maximum_weight=0.28,
        smoothing=0.70,
        **fedavg_arguments,
    ):
        super().__init__(**fedavg_arguments)
        self.model = evaluation_model
        self.x_utility = x_utility
        self.y_utility = y_utility
        self.shap_background = shap_background
        self.shap_samples = shap_samples
        self.shap_labels = shap_labels
        self.feature_names = list(feature_names)
        self.class_names = list(class_names)
        self.num_classes = len(self.class_names)
        self.result_dir = Path(result_dir)
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
        self.history = []

    def _predict(self, parameters):
        self.model.set_weights(parameters)
        probabilities = self.model.predict(
            self.x_utility,
            batch_size=2048,
            verbose=0,
        )
        return probabilities.argmax(axis=1)

    def _class_f1(self, parameters):
        return class_f1(
            self.y_utility,
            self._predict(parameters),
            self.num_classes,
        )

    def _server_shap(self):
        self.model.set_weights(self.global_parameters)
        matrix, _ = class_specific_mean_abs_shap(
            self.model,
            self.shap_background,
            self.shap_samples,
            self.shap_labels,
            self.num_classes,
            len(self.feature_names),
        )
        return matrix

    def _read_client_shap(self, metrics):
        matrix = np.zeros(
            (self.num_classes, len(self.feature_names)),
            dtype=np.float64,
        )
        available = np.zeros(self.num_classes, dtype=np.float64)
        for class_id, class_name in enumerate(self.class_names):
            available[class_id] = float(
                metrics.get(f"shap_available::{class_name}", 0.0)
            )
            for feature_id, feature_name in enumerate(self.feature_names):
                key = f"shap::{class_name}::{feature_name}"
                matrix[class_id, feature_id] = float(
                    metrics.get(key, 0.0)
                )
        if not np.all(np.isfinite(matrix)):
            raise ValueError("A client returned non-finite SHAP values.")
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
            records.append({
                "client_id": client_id,
                "num_examples": int(fit_result.num_examples),
                "parameters": parameters_to_ndarrays(
                    fit_result.parameters
                ),
                "shap": shap_matrix,
                "available": available,
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
            global_class_f1 = self._class_f1(self.global_parameters)
            client_class_f1 = np.stack([
                self._class_f1(item["parameters"])
                for item in records
            ])
            server_shap = self._server_shap()
            final_weights, diagnostic = calculate_client_weights(
                size_weights=size_weights,
                client_shap=np.stack([
                    item["shap"] for item in records
                ]),
                server_shap=server_shap,
                available=np.stack([
                    item["available"] for item in records
                ]),
                client_class_f1=client_class_f1,
                global_class_f1=global_class_f1,
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
        for index, item in enumerate(records):
            row = {
                "round": int(server_round),
                "client_id": item["client_id"],
                "num_examples": item["num_examples"],
                "fedavg_weight": float(size_weights[index]),
                "final_weight": float(final_weights[index]),
                "warmup": int(server_round <= self.warmup_rounds),
            }
            if diagnostic is not None:
                row["client_score"] = float(
                    diagnostic["score"][index]
                )
                row["raw_shap_weight"] = float(
                    diagnostic["shap_weight"][index]
                )
                for class_id, class_name in enumerate(self.class_names):
                    row[f"importance::{class_name}"] = float(
                        diagnostic["importance"][class_id]
                    )
                    row[f"reliability::{class_name}"] = float(
                        diagnostic["reliability"][index, class_id]
                    )
                    row[f"similarity::{class_name}"] = float(
                        diagnostic["similarity"][index, class_id]
                    )
                    row[f"gain::{class_name}"] = float(
                        diagnostic["gain"][index, class_id]
                    )
            self.history.append(row)

        pd.DataFrame(self.history).to_csv(
            self.result_dir / "aggregation_weights.csv",
            index=False,
        )
        print(
            f"[Class-specific Reliable SHAP] R{server_round}: "
            + ", ".join(
                f"C{item['client_id']}={weight:.4f}"
                for item, weight in zip(records, final_weights)
            )
        )
        return ndarrays_to_parameters(aggregated), {}
