"""Final-round post-hoc SHAP analysis for the server and every client.

The aggregation strategy calls :func:`save_final_posthoc_shap_analysis`
after the final client updates have been aggregated. The input SHAP matrices
must have shape ``[class, feature]`` and contain non-negative mean-absolute
SHAP values. Rows are normalised inside this module before comparison so the
analysis measures feature reliance rather than raw attribution scale.
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd


EPSILON = 1e-12


def _normalise_rows(values: np.ndarray) -> np.ndarray:
    values = np.asarray(values, dtype=np.float64)
    norms = np.linalg.norm(values, axis=1, keepdims=True)
    return np.divide(
        values,
        np.maximum(norms, EPSILON),
        out=np.zeros_like(values),
        where=norms > EPSILON,
    )


def _rank_descending(values: np.ndarray) -> np.ndarray:
    """Return one-based ranks, with rank 1 assigned to the largest value."""
    order = np.argsort(-np.asarray(values), kind="stable")
    ranks = np.empty(len(order), dtype=np.int64)
    ranks[order] = np.arange(1, len(order) + 1)
    return ranks


def _safe_correlation(left: np.ndarray, right: np.ndarray) -> float:
    left = np.asarray(left, dtype=np.float64)
    right = np.asarray(right, dtype=np.float64)
    if np.std(left) <= EPSILON or np.std(right) <= EPSILON:
        return 0.0
    return float(np.corrcoef(left, right)[0, 1])


def _cosine(left: np.ndarray, right: np.ndarray) -> float:
    denominator = np.linalg.norm(left) * np.linalg.norm(right)
    if denominator <= EPSILON:
        return 0.0
    return float(np.dot(left, right) / denominator)


def _top_indices(values: np.ndarray, count: int) -> np.ndarray:
    count = min(int(count), len(values))
    return np.argsort(-np.asarray(values), kind="stable")[:count]


def _top_overlap(
    left: np.ndarray,
    right: np.ndarray,
    count: int,
) -> tuple[int, float]:
    left_set = set(_top_indices(left, count).tolist())
    right_set = set(_top_indices(right, count).tolist())
    intersection = len(left_set & right_set)
    union = len(left_set | right_set)
    return intersection, float(intersection / union) if union else 0.0


def _validate_inputs(
    server_shap: np.ndarray,
    client_records: list[dict],
    class_names: list[str],
    feature_names: list[str],
) -> tuple[np.ndarray, list[dict]]:
    expected = (len(class_names), len(feature_names))
    server = np.asarray(server_shap, dtype=np.float64)
    if server.shape != expected:
        raise ValueError(
            f"Server SHAP shape {server.shape}; expected {expected}."
        )
    if not np.all(np.isfinite(server)):
        raise ValueError("Server SHAP contains non-finite values.")
    if np.any(server < 0):
        raise ValueError("Expected non-negative mean-absolute server SHAP.")

    validated = []
    seen_ids = set()
    for record in client_records:
        client_id = int(record["client_id"])
        if client_id in seen_ids:
            raise ValueError(f"Duplicate client ID in post-hoc data: {client_id}")
        seen_ids.add(client_id)

        matrix = np.asarray(record["shap"], dtype=np.float64)
        available = np.asarray(record["available"], dtype=np.float64)
        if matrix.shape != expected:
            raise ValueError(
                f"Client {client_id} SHAP shape {matrix.shape}; "
                f"expected {expected}."
            )
        if available.shape != (len(class_names),):
            raise ValueError(
                f"Client {client_id} availability shape {available.shape}."
            )
        if not np.all(np.isfinite(matrix)):
            raise ValueError(f"Client {client_id} SHAP is non-finite.")
        if np.any(matrix < 0):
            raise ValueError(
                f"Client {client_id}: expected non-negative mean-absolute SHAP."
            )
        validated.append({
            "client_id": client_id,
            "shap": matrix,
            "available": (available > 0.5).astype(np.int64),
        })
    validated.sort(key=lambda item: item["client_id"])
    return server, validated


def save_local_client_posthoc_shap(
    output_dir: str | Path,
    *,
    server_round: int,
    client_id: int,
    class_names: Iterable[str],
    feature_names: Iterable[str],
    client_shap: np.ndarray,
    available: np.ndarray,
) -> Path:
    """Save the final client's own class-feature SHAP table locally."""
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    class_names = list(class_names)
    feature_names = list(feature_names)
    matrix = _normalise_rows(np.asarray(client_shap, dtype=np.float64))
    available = (np.asarray(available) > 0.5).astype(np.int64)

    if matrix.shape != (len(class_names), len(feature_names)):
        raise ValueError(f"Unexpected local client SHAP shape {matrix.shape}.")

    rows = []
    for class_id, class_name in enumerate(class_names):
        ranks = _rank_descending(matrix[class_id])
        for feature_id, feature_name in enumerate(feature_names):
            rows.append({
                "round": int(server_round),
                "client_id": int(client_id),
                "class_id": class_id,
                "class_name": class_name,
                "available": int(available[class_id]),
                "feature_id": feature_id,
                "feature_name": feature_name,
                "normalized_mean_abs_shap": float(
                    matrix[class_id, feature_id]
                ),
                "feature_rank_within_class": int(ranks[feature_id]),
            })
    path = output_dir / f"final_client_{int(client_id)}_shap_by_feature.csv"
    pd.DataFrame(rows).to_csv(path, index=False)
    return path


def save_final_posthoc_shap_analysis(
    result_dir: str | Path,
    *,
    server_round: int,
    feature_names: Iterable[str],
    class_names: Iterable[str],
    server_shap: np.ndarray,
    client_records: list[dict],
) -> dict[str, Path]:
    """Write final server/client SHAP tables and comparison diagnostics."""
    result_dir = Path(result_dir)
    result_dir.mkdir(parents=True, exist_ok=True)
    feature_names = list(feature_names)
    class_names = list(class_names)
    server, clients = _validate_inputs(
        server_shap,
        client_records,
        class_names,
        feature_names,
    )
    server = _normalise_rows(server)
    for client in clients:
        client["shap"] = _normalise_rows(client["shap"])

    server_rows = []
    client_rows = []
    comparison_rows = []
    by_feature_rows = []
    top_feature_rows = []

    for class_id, class_name in enumerate(class_names):
        server_vector = server[class_id]
        server_ranks = _rank_descending(server_vector)
        for feature_id, feature_name in enumerate(feature_names):
            server_rows.append({
                "round": int(server_round),
                "source": "server",
                "class_id": class_id,
                "class_name": class_name,
                "feature_id": feature_id,
                "feature_name": feature_name,
                "normalized_mean_abs_shap": float(server_vector[feature_id]),
                "feature_rank_within_class": int(server_ranks[feature_id]),
            })
        for rank, feature_id in enumerate(
            _top_indices(server_vector, 10),
            start=1,
        ):
            top_feature_rows.append({
                "round": int(server_round),
                "source": "server",
                "client_id": "",
                "class_name": class_name,
                "rank": rank,
                "feature_name": feature_names[feature_id],
                "normalized_mean_abs_shap": float(
                    server_vector[feature_id]
                ),
            })

        for client in clients:
            client_id = client["client_id"]
            available = int(client["available"][class_id])
            client_vector = client["shap"][class_id]
            client_ranks = _rank_descending(client_vector)

            for feature_id, feature_name in enumerate(feature_names):
                client_rows.append({
                    "round": int(server_round),
                    "source": f"client_{client_id}",
                    "client_id": client_id,
                    "class_id": class_id,
                    "class_name": class_name,
                    "available": available,
                    "feature_id": feature_id,
                    "feature_name": feature_name,
                    "normalized_mean_abs_shap": float(
                        client_vector[feature_id]
                    ),
                    "feature_rank_within_class": int(
                        client_ranks[feature_id]
                    ),
                })
                by_feature_rows.append({
                    "round": int(server_round),
                    "client_id": client_id,
                    "class_name": class_name,
                    "available": available,
                    "feature_name": feature_name,
                    "server_normalized_mean_abs_shap": float(
                        server_vector[feature_id]
                    ),
                    "client_normalized_mean_abs_shap": float(
                        client_vector[feature_id]
                    ),
                    "client_minus_server": float(
                        client_vector[feature_id]
                        - server_vector[feature_id]
                    ),
                    "absolute_difference": float(abs(
                        client_vector[feature_id]
                        - server_vector[feature_id]
                    )),
                    "server_rank": int(server_ranks[feature_id]),
                    "client_rank": int(client_ranks[feature_id]),
                })

            if not available:
                comparison_rows.append({
                    "round": int(server_round),
                    "client_id": client_id,
                    "class_name": class_name,
                    "available": 0,
                    "cosine_similarity": np.nan,
                    "pearson_correlation": np.nan,
                    "spearman_rank_correlation": np.nan,
                    "mean_absolute_difference": np.nan,
                    "top_server_feature": feature_names[
                        int(np.argmax(server_vector))
                    ],
                    "top_client_feature": "",
                    "top1_match": np.nan,
                    "top5_overlap_count": np.nan,
                    "top5_jaccard": np.nan,
                    "top10_overlap_count": np.nan,
                    "top10_jaccard": np.nan,
                })
                continue

            server_rank_vector = _rank_descending(server_vector)
            client_rank_vector = _rank_descending(client_vector)
            top5_count, top5_jaccard = _top_overlap(
                server_vector,
                client_vector,
                5,
            )
            top10_count, top10_jaccard = _top_overlap(
                server_vector,
                client_vector,
                10,
            )
            top_server = int(np.argmax(server_vector))
            top_client = int(np.argmax(client_vector))
            comparison_rows.append({
                "round": int(server_round),
                "client_id": client_id,
                "class_name": class_name,
                "available": 1,
                "cosine_similarity": _cosine(
                    server_vector,
                    client_vector,
                ),
                "pearson_correlation": _safe_correlation(
                    server_vector,
                    client_vector,
                ),
                "spearman_rank_correlation": _safe_correlation(
                    server_rank_vector,
                    client_rank_vector,
                ),
                "mean_absolute_difference": float(np.mean(np.abs(
                    server_vector - client_vector
                ))),
                "top_server_feature": feature_names[top_server],
                "top_client_feature": feature_names[top_client],
                "top1_match": int(top_server == top_client),
                "top5_overlap_count": top5_count,
                "top5_jaccard": top5_jaccard,
                "top10_overlap_count": top10_count,
                "top10_jaccard": top10_jaccard,
            })
            for rank, feature_id in enumerate(
                _top_indices(client_vector, 10),
                start=1,
            ):
                top_feature_rows.append({
                    "round": int(server_round),
                    "source": f"client_{client_id}",
                    "client_id": client_id,
                    "class_name": class_name,
                    "rank": rank,
                    "feature_name": feature_names[feature_id],
                    "normalized_mean_abs_shap": float(
                        client_vector[feature_id]
                    ),
                })

    server_frame = pd.DataFrame(server_rows)
    client_frame = pd.DataFrame(client_rows)
    comparison_frame = pd.DataFrame(comparison_rows)
    by_feature_frame = pd.DataFrame(by_feature_rows)
    top_feature_frame = pd.DataFrame(top_feature_rows)

    available_comparisons = comparison_frame.loc[
        comparison_frame["available"] == 1
    ].copy()
    client_summary = (
        available_comparisons.groupby("client_id", as_index=False)
        .agg(
            available_classes=("class_name", "count"),
            mean_cosine_similarity=("cosine_similarity", "mean"),
            minimum_cosine_similarity=("cosine_similarity", "min"),
            mean_pearson_correlation=("pearson_correlation", "mean"),
            mean_spearman_rank_correlation=(
                "spearman_rank_correlation",
                "mean",
            ),
            mean_absolute_difference=(
                "mean_absolute_difference",
                "mean",
            ),
            mean_top5_jaccard=("top5_jaccard", "mean"),
            top1_match_rate=("top1_match", "mean"),
        )
    )

    class_summary_rows = []
    for class_name in class_names:
        rows = available_comparisons.loc[
            available_comparisons["class_name"] == class_name
        ]
        if rows.empty:
            continue
        best_index = rows["cosine_similarity"].idxmax()
        worst_index = rows["cosine_similarity"].idxmin()
        class_summary_rows.append({
            "round": int(server_round),
            "class_name": class_name,
            "available_clients": int(len(rows)),
            "mean_cosine_similarity": float(
                rows["cosine_similarity"].mean()
            ),
            "minimum_cosine_similarity": float(
                rows["cosine_similarity"].min()
            ),
            "maximum_cosine_similarity": float(
                rows["cosine_similarity"].max()
            ),
            "best_aligned_client": int(
                rows.loc[best_index, "client_id"]
            ),
            "worst_aligned_client": int(
                rows.loc[worst_index, "client_id"]
            ),
            "mean_absolute_difference": float(
                rows["mean_absolute_difference"].mean()
            ),
            "mean_top5_jaccard": float(rows["top5_jaccard"].mean()),
        })
    class_summary = pd.DataFrame(class_summary_rows)

    feature_summary_rows = []
    for feature_id, feature_name in enumerate(feature_names):
        row = {
            "round": int(server_round),
            "feature_name": feature_name,
            "server_mean_across_classes": float(
                server[:, feature_id].mean()
            ),
        }
        available_client_means = []
        for client in clients:
            mask = client["available"].astype(bool)
            value = (
                float(client["shap"][mask, feature_id].mean())
                if np.any(mask)
                else np.nan
            )
            row[f"client_{client['client_id']}_mean_available_classes"] = value
            if np.isfinite(value):
                available_client_means.append(value)
        row["mean_across_clients"] = float(
            np.mean(available_client_means)
        )
        row["client_mean_minus_server"] = float(
            row["mean_across_clients"]
            - row["server_mean_across_classes"]
        )
        row["absolute_client_server_difference"] = abs(
            row["client_mean_minus_server"]
        )
        feature_summary_rows.append(row)
    feature_summary = pd.DataFrame(feature_summary_rows).sort_values(
        "server_mean_across_classes",
        ascending=False,
    )

    paths = {
        "server_shap": result_dir
        / "final_posthoc_server_shap_by_feature.csv",
        "client_shap": result_dir
        / "final_posthoc_client_shap_by_feature.csv",
        "by_feature_comparison": result_dir
        / "final_posthoc_server_client_by_feature.csv",
        "class_client_comparison": result_dir
        / "final_posthoc_server_client_comparison.csv",
        "client_summary": result_dir
        / "final_posthoc_client_summary.csv",
        "class_summary": result_dir
        / "final_posthoc_class_summary.csv",
        "global_feature_summary": result_dir
        / "final_posthoc_global_feature_summary.csv",
        "top_features": result_dir
        / "final_posthoc_top_features.csv",
        "manifest": result_dir / "final_posthoc_manifest.json",
    }
    server_frame.to_csv(paths["server_shap"], index=False)
    client_frame.to_csv(paths["client_shap"], index=False)
    by_feature_frame.to_csv(paths["by_feature_comparison"], index=False)
    comparison_frame.to_csv(
        paths["class_client_comparison"],
        index=False,
    )
    client_summary.to_csv(paths["client_summary"], index=False)
    class_summary.to_csv(paths["class_summary"], index=False)
    feature_summary.to_csv(
        paths["global_feature_summary"],
        index=False,
    )
    top_feature_frame.to_csv(paths["top_features"], index=False)

    manifest = {
        "round": int(server_round),
        "description": (
            "Cross-distribution post-hoc comparison of the final aggregated "
            "server model explained on the untouched server test partition "
            "and each round-10 local client model explained on that client's "
            "local test partition."
        ),
        "server_shap_data": (
            "Balanced samples from the untouched server final-test partition."
        ),
        "client_shap_data": (
            "Balanced samples from each client's local test partition, "
            "calculated after that client's round-10 local training."
        ),
        "shap_value_definition": (
            "Per-class mean absolute GradientExplainer SHAP, L2-normalised "
            "within each class before comparison."
        ),
        "important_interpretation": (
            "High similarity means the server and client rely on a similar "
            "feature pattern on their respective test distributions. "
            "Differences can arise from both model differences and non-IID "
            "data differences; similarity alone does not prove accuracy."
        ),
        "classes": class_names,
        "features": feature_names,
        "clients": [client["client_id"] for client in clients],
        "files": {
            name: path.name
            for name, path in paths.items()
            if name != "manifest"
        },
    }
    paths["manifest"].write_text(
        json.dumps(manifest, indent=2),
        encoding="utf-8",
    )
    return paths
