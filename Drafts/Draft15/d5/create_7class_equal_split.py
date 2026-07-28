"""Repartition the existing seven-class client pool into equal-size clients.

The server holdout and scaler are intentionally unchanged. All client-pool
rows are assigned exactly once. Client sizes are exact (or differ by one row
when the total is not divisible by five), while class proportions follow a
capacity-constrained Dirichlet distribution.
"""
from pathlib import Path

import numpy as np
import pandas as pd

from Labels7 import ALL_LABELS, NUM_CLASSES, NUM_FEATURES

INPUT_TEMPLATE = r"D:\CAPSTONE\Dir7_Client{client_id}.csv"
OUTPUT_TEMPLATE = r"D:\CAPSTONE\Dir7Equal_Client{client_id}.csv"
SUMMARY_PATH = r"D:\CAPSTONE\Dir7Equal_Distribution.csv"

NUM_CLIENTS = 5
ALPHA = 0.5
MIN_ROWS_PER_CLASS_PER_CLIENT = 20
SEED = 42
MAX_IPF_ITERATIONS = 10_000
IPF_TOLERANCE = 1e-7

rng = np.random.RandomState(SEED)


def load_existing_client_pool():
    frames = []
    for client_id in range(1, NUM_CLIENTS + 1):
        path = INPUT_TEMPLATE.format(client_id=client_id)
        print(f"Loading {path}...")
        frame = pd.read_csv(path)
        frame.drop(
            columns=[
                column
                for column in frame.columns
                if column.lower().startswith("unnamed:")
            ],
            inplace=True,
            errors="ignore",
        )
        frames.append(frame)

    pool = pd.concat(frames, ignore_index=True)
    pool.replace([np.inf, -np.inf], np.nan, inplace=True)
    if pool.isna().any().any():
        raise ValueError(
            "The existing client pool contains missing or infinite values. "
            "Do not drop rows here because every source row must be preserved."
        )

    if "Label" not in pool.columns:
        raise ValueError("Label column is missing.")

    pool["Label"] = (
        pool["Label"].astype(str).str.strip().str.upper()
    )
    unexpected = sorted(set(pool["Label"]) - set(ALL_LABELS))
    if unexpected:
        raise ValueError(f"Unexpected seven-class labels: {unexpected}")

    feature_columns = [
        column for column in pool.columns
        if column != "Label"
    ]
    if len(feature_columns) != NUM_FEATURES:
        raise ValueError(
            f"Expected {NUM_FEATURES} features, "
            f"found {len(feature_columns)}."
        )
    return pool


def target_client_sizes(total_rows):
    base, remainder = divmod(total_rows, NUM_CLIENTS)
    targets = np.full(NUM_CLIENTS, base, dtype=np.int64)
    targets[:remainder] += 1
    return targets


def iterative_proportional_fit(
    row_totals,
    column_totals,
    preferences,
):
    """Produce a positive real matrix with requested row/column margins."""
    matrix = np.maximum(preferences, 1e-12).astype(np.float64)

    for _ in range(MAX_IPF_ITERATIONS):
        matrix *= (
            row_totals / np.maximum(matrix.sum(axis=1), 1e-12)
        )[:, None]
        matrix *= (
            column_totals / np.maximum(matrix.sum(axis=0), 1e-12)
        )[None, :]

        row_error = np.max(np.abs(matrix.sum(axis=1) - row_totals))
        column_error = np.max(
            np.abs(matrix.sum(axis=0) - column_totals)
        )
        if max(row_error, column_error) <= IPF_TOLERANCE:
            return matrix

    raise RuntimeError("Capacity-constrained Dirichlet IPF did not converge.")


def integerize_matrix(real_matrix, row_totals, column_totals):
    """Round while retaining both margins exactly."""
    integer_matrix = np.floor(real_matrix).astype(np.int64)
    fractional = real_matrix - integer_matrix
    row_remaining = row_totals - integer_matrix.sum(axis=1)
    column_remaining = column_totals - integer_matrix.sum(axis=0)

    while row_remaining.sum() > 0:
        available = (
            (row_remaining[:, None] > 0)
            & (column_remaining[None, :] > 0)
        )
        if not np.any(available):
            raise RuntimeError("Could not complete integer margin rounding.")

        scores = np.where(available, fractional, -1.0)
        class_index, client_index = np.unravel_index(
            np.argmax(scores),
            scores.shape,
        )
        integer_matrix[class_index, client_index] += 1
        row_remaining[class_index] -= 1
        column_remaining[client_index] -= 1
        fractional[class_index, client_index] = 0.0

    if not np.array_equal(integer_matrix.sum(axis=1), row_totals):
        raise RuntimeError("Class totals changed during integerization.")
    if not np.array_equal(integer_matrix.sum(axis=0), column_totals):
        raise RuntimeError("Client capacities changed during integerization.")
    return integer_matrix


def build_allocation(class_totals, client_targets):
    minimum = MIN_ROWS_PER_CLASS_PER_CLIENT
    minimum_matrix = np.full(
        (NUM_CLASSES, NUM_CLIENTS),
        minimum,
        dtype=np.int64,
    )

    if np.any(class_totals < minimum * NUM_CLIENTS):
        insufficient = {
            ALL_LABELS[index]: int(total)
            for index, total in enumerate(class_totals)
            if total < minimum * NUM_CLIENTS
        }
        raise ValueError(
            "Minimum allocation is infeasible for classes: "
            f"{insufficient}"
        )

    residual_rows = class_totals - minimum_matrix.sum(axis=1)
    residual_columns = (
        client_targets - minimum_matrix.sum(axis=0)
    )
    if np.any(residual_columns < 0):
        raise ValueError("Client targets are smaller than reserved minima.")

    preferences = np.stack([
        rng.dirichlet(np.full(NUM_CLIENTS, ALPHA))
        for _ in range(NUM_CLASSES)
    ])
    real_matrix = iterative_proportional_fit(
        residual_rows.astype(np.float64),
        residual_columns.astype(np.float64),
        preferences,
    )
    residual_integer = integerize_matrix(
        real_matrix,
        residual_rows,
        residual_columns,
    )
    return minimum_matrix + residual_integer


def main():
    pool = load_existing_client_pool()
    total_rows = len(pool)
    client_targets = target_client_sizes(total_rows)

    class_indices = []
    class_totals = []
    for class_name in ALL_LABELS:
        indices = np.where(pool["Label"].to_numpy() == class_name)[0]
        rng.shuffle(indices)
        class_indices.append(indices)
        class_totals.append(len(indices))
    class_totals = np.asarray(class_totals, dtype=np.int64)

    allocation = build_allocation(class_totals, client_targets)
    client_indices = [[] for _ in range(NUM_CLIENTS)]

    for class_index, indices in enumerate(class_indices):
        start = 0
        for client_index in range(NUM_CLIENTS):
            count = int(allocation[class_index, client_index])
            end = start + count
            client_indices[client_index].extend(
                indices[start:end].tolist()
            )
            start = end
        if start != len(indices):
            raise RuntimeError(
                f"Class {ALL_LABELS[class_index]} was not fully allocated."
            )

    assigned = np.asarray(
        [index for indices in client_indices for index in indices],
        dtype=np.int64,
    )
    if len(assigned) != total_rows:
        raise RuntimeError(
            f"Assigned {len(assigned)} rows, expected {total_rows}."
        )
    if len(np.unique(assigned)) != total_rows:
        raise RuntimeError("Duplicate client-pool rows were assigned.")
    if assigned.min() != 0 or assigned.max() != total_rows - 1:
        raise RuntimeError("At least one client-pool row was omitted.")

    summary_rows = []
    for client_index in range(NUM_CLIENTS):
        indices = np.asarray(
            client_indices[client_index],
            dtype=np.int64,
        )
        rng.shuffle(indices)
        client_frame = pool.iloc[indices].reset_index(drop=True)
        expected = int(client_targets[client_index])
        if len(client_frame) != expected:
            raise RuntimeError(
                f"Client {client_index + 1} has {len(client_frame)} rows, "
                f"expected {expected}."
            )

        output_path = OUTPUT_TEMPLATE.format(
            client_id=client_index + 1
        )
        client_frame.to_csv(output_path, index=False)
        print(
            f"Client {client_index + 1}: {len(client_frame):,} rows "
            f"saved to {output_path}"
        )

        counts = client_frame["Label"].value_counts()
        for class_name in ALL_LABELS:
            summary_rows.append({
                "client": f"Client{client_index + 1}",
                "label": class_name,
                "count": int(counts.get(class_name, 0)),
            })

    summary = pd.DataFrame(summary_rows)
    pivot = summary.pivot(
        index="label",
        columns="client",
        values="count",
    ).fillna(0).astype(int)
    client_columns = [
        f"Client{client_id}"
        for client_id in range(1, NUM_CLIENTS + 1)
    ]
    pivot["present_in"] = (pivot[client_columns] > 0).sum(axis=1)
    pivot["heterogeneity_score"] = (
        pivot[client_columns].std(axis=1)
        / (pivot[client_columns].mean(axis=1) + 1.0)
    )
    pivot.to_csv(SUMMARY_PATH)

    print("\n" + "=" * 100)
    print(
        f"EQUAL-SIZE SEVEN-CLASS LABEL-SKEW SPLIT "
        f"| alpha={ALPHA} | clients={NUM_CLIENTS}"
    )
    print("=" * 100)
    print(pivot[client_columns + [
        "present_in",
        "heterogeneity_score",
    ]].to_string())
    print(f"\nDistribution saved to {SUMMARY_PATH}")
    print(f"Client targets: {client_targets.tolist()}")


if __name__ == "__main__":
    main()
