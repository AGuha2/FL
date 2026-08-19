"""Create a controlled, equal-size, six-class non-IID split.

DDoS and DoS are independently capped to the Mirai population before the
stratified server holdout. Every retained row is then assigned exactly once
to either the holdout or one of five equal-size specialist clients.
"""
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import RobustScaler

from Labels6 import ALL_LABELS, NUM_FEATURES, convert_series_to_family


SOURCE_FILES = [rf"D:\CAPSTONE\Merged0{i}.csv" for i in range(1, 6)]
SERVER_TEST_PATH = Path(r"D:\CAPSTONE\IoT6Bal_Server.csv")
SCALER_PATH = Path(r"D:\CAPSTONE\IoT6Bal_Scaler.pkl")
CLIENT_PATH_TEMPLATE = r"D:\CAPSTONE\IoT6Bal_Client{client_id}.csv"
SUMMARY_PATH = Path(r"D:\CAPSTONE\IoT6Bal_Distribution.csv")
EXCLUSION_SUMMARY_PATH = Path(r"D:\CAPSTONE\IoT6Bal_Excluded.csv")
DOWNSAMPLING_SUMMARY_PATH = Path(
    r"D:\CAPSTONE\IoT6Bal_Downsampling.csv"
)

NUM_CLIENTS = 5
TEST_FRACTION = 0.10
SEED = 42
SPECIALIST_STRENGTH = 20.0
SHARED_STRENGTH = 1.0
MIN_ROWS_PER_FAMILY_PER_CLIENT = 25
MAX_IPF_ITERATIONS = 10_000
IPF_TOLERANCE = 1e-7

SPECIALISTS = {
    "DDOS": 0,
    "DOS": 1,
    "MIRAI": 2,
    "RECON": 3,
    "OTHER": 4,
}


def target_client_sizes(total_rows):
    base, remainder = divmod(total_rows, NUM_CLIENTS)
    targets = np.full(NUM_CLIENTS, base, dtype=np.int64)
    targets[:remainder] += 1
    return targets


def iterative_proportional_fit(row_totals, column_totals, preferences):
    matrix = np.maximum(preferences, 1e-12).astype(np.float64)
    for _ in range(MAX_IPF_ITERATIONS):
        matrix *= (row_totals / np.maximum(matrix.sum(axis=1), 1e-12))[:, None]
        matrix *= (column_totals / np.maximum(matrix.sum(axis=0), 1e-12))[None, :]
        error = max(
            np.max(np.abs(matrix.sum(axis=1) - row_totals)),
            np.max(np.abs(matrix.sum(axis=0) - column_totals)),
        )
        if error <= IPF_TOLERANCE:
            return matrix
    raise RuntimeError("Specialist allocation IPF did not converge.")


def integerize_matrix(real_matrix, row_totals, column_totals):
    result = np.floor(real_matrix).astype(np.int64)
    fractions = real_matrix - result
    row_remaining = row_totals - result.sum(axis=1)
    column_remaining = column_totals - result.sum(axis=0)
    while row_remaining.sum() > 0:
        available = ((row_remaining[:, None] > 0)
                     & (column_remaining[None, :] > 0))
        if not np.any(available):
            raise RuntimeError("Could not retain exact allocation margins.")
        # Ineligible cells must never tie with previously selected cells.
        # Using -inf prevents np.argmax from choosing an exhausted row/column.
        scores = np.where(available, fractions, -np.inf)
        row, column = np.unravel_index(np.argmax(scores), scores.shape)
        result[row, column] += 1
        row_remaining[row] -= 1
        column_remaining[column] -= 1
        # A cell may be reused if the greedy choices otherwise reach a
        # transportation-rounding dead end. Zero keeps it below positive
        # fractional candidates while preserving exact row/column margins.
        fractions[row, column] = 0.0
    if not np.array_equal(result.sum(axis=1), row_totals):
        raise RuntimeError("A family total changed during integerization.")
    if not np.array_equal(result.sum(axis=0), column_totals):
        raise RuntimeError("A client size changed during integerization.")
    return result


def build_allocation(family_totals, client_targets):
    minimum = np.full(
        (len(ALL_LABELS), NUM_CLIENTS),
        MIN_ROWS_PER_FAMILY_PER_CLIENT,
        dtype=np.int64,
    )
    if np.any(family_totals < minimum.sum(axis=1)):
        raise ValueError("A family is too small for the requested shared minimum.")
    residual_rows = family_totals - minimum.sum(axis=1)
    residual_columns = client_targets - minimum.sum(axis=0)

    preferences = np.full(
        (len(ALL_LABELS), NUM_CLIENTS),
        SHARED_STRENGTH,
        dtype=np.float64,
    )
    for family, client_index in SPECIALISTS.items():
        preferences[ALL_LABELS.index(family), client_index] = SPECIALIST_STRENGTH

    fitted = iterative_proportional_fit(
        residual_rows.astype(np.float64),
        residual_columns.astype(np.float64),
        preferences,
    )
    return minimum + integerize_matrix(
        fitted, residual_rows, residual_columns
    )


def load_and_prepare_sources():
    print("Loading original CICIoT2023 CSV files...")
    frame = pd.concat([pd.read_csv(path) for path in SOURCE_FILES], ignore_index=True)
    frame.replace([np.inf, -np.inf], np.nan, inplace=True)
    frame.dropna(inplace=True)
    frame.drop(
        columns=[c for c in frame.columns if c.lower().startswith("unnamed:")],
        inplace=True,
        errors="ignore",
    )
    if "Label" not in frame.columns:
        raise ValueError("Label column is missing.")
    original_labels = frame["Label"].astype(str).str.strip().str.upper()
    mapped_labels = convert_series_to_family(original_labels)
    excluded = original_labels[mapped_labels.isna()].value_counts().sort_index()
    excluded.rename_axis("original_label").rename("excluded_rows").to_csv(
        EXCLUSION_SUMMARY_PATH
    )
    keep = mapped_labels.notna()
    frame = frame.loc[keep].copy()
    frame["Label"] = mapped_labels.loc[keep].to_numpy()

    # Independently cap DDoS and DoS to the Mirai count. Sampling occurs
    # before the holdout so server and clients share the capped population.
    rng = np.random.RandomState(SEED)
    mirai_count = int((frame["Label"] == "MIRAI").sum())
    retained_parts = [
        frame.index[~frame["Label"].isin(["DDOS", "DOS"])].to_numpy()
    ]
    downsampling_rows = []
    for family in ("DDOS", "DOS"):
        family_indices = frame.index[frame["Label"] == family].to_numpy()
        original_count = len(family_indices)
        if original_count < mirai_count:
            raise ValueError(
                f"{family} has {original_count:,} rows, fewer than "
                f"MIRAI ({mirai_count:,}); it cannot be capped upward."
            )
        retained_parts.append(
            rng.choice(family_indices, size=mirai_count, replace=False)
        )
        downsampling_rows.append({
            "family": family,
            "original_rows": original_count,
            "cap_reference_family": "MIRAI",
            "retained_rows": mirai_count,
            "downsampled_rows": original_count - mirai_count,
            "seed": SEED,
        })
    retained_indices = np.concatenate(retained_parts)
    rng.shuffle(retained_indices)
    frame = frame.loc[retained_indices].copy()

    pd.DataFrame(downsampling_rows).to_csv(
        DOWNSAMPLING_SUMMARY_PATH,
        index=False,
    )
    frame.reset_index(drop=True, inplace=True)
    feature_columns = [c for c in frame.columns if c != "Label"]
    if len(feature_columns) != NUM_FEATURES:
        raise ValueError(f"Expected {NUM_FEATURES} features; found {len(feature_columns)}.")
    print(f"Excluded/unmapped rows: {int(excluded.sum()):,}")
    print(f"Exclusion summary -> {EXCLUSION_SUMMARY_PATH}")
    print(
        f"Capped DDOS and DOS independently to {mirai_count:,} rows "
        "each (equal to MIRAI)."
    )
    print(f"Downsampling summary -> {DOWNSAMPLING_SUMMARY_PATH}")
    return frame, feature_columns


def main():
    frame, feature_columns = load_and_prepare_sources()

    # Deliberately unchanged from the original seven-class splitter.
    client_pool, server_test = train_test_split(
        frame,
        test_size=TEST_FRACTION,
        random_state=SEED,
        stratify=frame["Label"],
    )
    client_pool = client_pool.reset_index(drop=True)
    server_test = server_test.reset_index(drop=True)
    server_test.to_csv(SERVER_TEST_PATH, index=False)
    scaler = RobustScaler()
    scaler.fit(client_pool[feature_columns])
    joblib.dump(scaler, SCALER_PATH)

    rng = np.random.RandomState(SEED)
    client_targets = target_client_sizes(len(client_pool))
    family_indices = []
    family_totals = []
    for family in ALL_LABELS:
        indices = np.flatnonzero(client_pool["Label"].to_numpy() == family)
        rng.shuffle(indices)
        family_indices.append(indices)
        family_totals.append(len(indices))
    family_totals = np.asarray(family_totals, dtype=np.int64)
    allocation = build_allocation(family_totals, client_targets)

    client_indices = [[] for _ in range(NUM_CLIENTS)]
    for family_index, indices in enumerate(family_indices):
        start = 0
        for client_index in range(NUM_CLIENTS):
            end = start + int(allocation[family_index, client_index])
            client_indices[client_index].extend(indices[start:end].tolist())
            start = end
        if start != len(indices):
            raise RuntimeError(f"Family {ALL_LABELS[family_index]} was not fully assigned.")

    assigned = np.asarray([i for group in client_indices for i in group])
    if len(assigned) != len(client_pool):
        raise RuntimeError("The assigned-row count does not equal the client pool.")
    if len(np.unique(assigned)) != len(client_pool):
        raise RuntimeError("A client-pool row was duplicated or omitted.")
    if assigned.min() != 0 or assigned.max() != len(client_pool) - 1:
        raise RuntimeError("A client-pool row was omitted.")

    summary = pd.DataFrame(index=ALL_LABELS)
    for client_index, indices in enumerate(client_indices):
        indices = np.asarray(indices, dtype=np.int64)
        rng.shuffle(indices)
        client_frame = client_pool.iloc[indices].reset_index(drop=True)
        expected = int(client_targets[client_index])
        if len(client_frame) != expected:
            raise RuntimeError(f"Client {client_index + 1} size is not {expected}.")
        path = CLIENT_PATH_TEMPLATE.format(client_id=client_index + 1)
        client_frame.to_csv(path, index=False)
        summary[f"Client{client_index + 1}"] = (
            client_frame["Label"].value_counts().reindex(ALL_LABELS, fill_value=0)
        )
        print(f"Client {client_index + 1}: {len(client_frame):,} rows -> {path}")

    summary["total"] = summary.sum(axis=1)
    summary["assigned_specialist"] = [
        f"Client{SPECIALISTS[family] + 1}"
        if family in SPECIALISTS else "shared"
        for family in ALL_LABELS
    ]
    summary.to_csv(SUMMARY_PATH, index_label="label")
    print(summary.to_string())
    print(f"Server holdout: {len(server_test):,} rows -> {SERVER_TEST_PATH}")
    print(f"Scaler -> {SCALER_PATH}")
    print(f"Distribution -> {SUMMARY_PATH}")


if __name__ == "__main__":
    main()
