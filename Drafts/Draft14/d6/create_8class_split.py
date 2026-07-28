import joblib
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import RobustScaler

from Labels import (
    le,
    NUM_CLASSES,
    NUM_FEATURES,
    convert_series_to_family,
)


SOURCE_FILES = [
    r"D:\CAPSTONE\Merged01.csv",
    r"D:\CAPSTONE\Merged02.csv",
    r"D:\CAPSTONE\Merged03.csv",
    r"D:\CAPSTONE\Merged04.csv",
    r"D:\CAPSTONE\Merged05.csv",
]

SERVER_TEST_PATH = r"D:\CAPSTONE\Server_Test_Class.csv"
SCALER_PATH = r"D:\CAPSTONE\global_scaler_Class.pkl"
CLIENT_PATH_TEMPLATE = r"D:\CAPSTONE\Dir_Client{client_id}.csv"
SUMMARY_PATH = r"D:\CAPSTONE\Dir_Distribution.csv"

NUM_CLIENTS = 5
ALPHA = 0.5
SEED = 42
TEST_FRACTION = 0.10

rng = np.random.RandomState(SEED)


print("Loading source CSV files...")

df = pd.concat(
    [pd.read_csv(path) for path in SOURCE_FILES],
    ignore_index=True,
)

df.replace([np.inf, -np.inf], np.nan, inplace=True)
df.dropna(inplace=True)
df.drop(
    columns=[
        column
        for column in df.columns
        if column.lower().startswith("unnamed:")
    ],
    inplace=True,
    errors="ignore",
)

if "Label" not in df.columns:
    raise ValueError("Label column is missing.")

df["Label"] = convert_series_to_family(df["Label"])
df.reset_index(drop=True, inplace=True)

print(f"Total rows loaded: {len(df):,}")
print(f"Mapped classes: {sorted(df['Label'].unique())}")


unexpected = set(df["Label"]) - set(le.classes_)

if unexpected:
    raise ValueError(
        f"Unexpected family labels: {sorted(unexpected)}"
    )


feature_cols = [
    column
    for column in df.columns
    if column != "Label"
]

if len(feature_cols) != NUM_FEATURES:
    raise ValueError(
        f"Expected {NUM_FEATURES} features, "
        f"but found {len(feature_cols)}.\n"
        f"Features: {feature_cols}"
    )


# ============================================================
# Separate the server test set before federated partitioning
# ============================================================

client_pool_df, server_test_df = train_test_split(
    df,
    test_size=TEST_FRACTION,
    random_state=SEED,
    stratify=df["Label"],
)

client_pool_df = client_pool_df.reset_index(drop=True)
server_test_df = server_test_df.reset_index(drop=True)

print(f"\nClient pool: {len(client_pool_df):,}")
print(f"Server test: {len(server_test_df):,}")

server_test_df.to_csv(
    SERVER_TEST_PATH,
    index=False,
)

print(f"Server test saved to {SERVER_TEST_PATH}")


# ============================================================
# Fit one shared scaler on the client training pool
# ============================================================

scaler = RobustScaler()
scaler.fit(client_pool_df[feature_cols])

joblib.dump(
    scaler,
    SCALER_PATH,
)

print(f"Scaler saved to {SCALER_PATH}")


# ============================================================
# Dirichlet non-IID split using the eight family labels
# ============================================================

classes = sorted(
    client_pool_df["Label"].unique()
)

if len(classes) != NUM_CLASSES:
    print(
        f"Warning: client pool contains {len(classes)} "
        f"of the expected {NUM_CLASSES} classes."
    )


class_indices = {
    class_name: client_pool_df.index[
        client_pool_df["Label"] == class_name
    ].to_numpy(copy=True)
    for class_name in classes
}


client_indices: list[list[int]] = [
    [] for _ in range(NUM_CLIENTS)
]


for class_name in classes:
    indices = class_indices[class_name].copy()
    rng.shuffle(indices)

    proportions = rng.dirichlet(
        np.full(NUM_CLIENTS, ALPHA)
    )

    split_sizes = np.floor(
        proportions * len(indices)
    ).astype(int)

    remainder = len(indices) - split_sizes.sum()

    if remainder > 0:
        fractional = (
            proportions * len(indices)
            - split_sizes
        )

        recipients = np.argsort(
            fractional
        )[::-1][:remainder]

        split_sizes[recipients] += 1

    start = 0

    for client_index, split_size in enumerate(split_sizes):
        end = start + int(split_size)

        client_indices[client_index].extend(
            indices[start:end].tolist()
        )

        start = end


# Ensure every row was assigned exactly once.
assigned_indices = [
    index
    for client in client_indices
    for index in client
]

if len(assigned_indices) != len(client_pool_df):
    raise RuntimeError(
        f"Assigned {len(assigned_indices)} rows, "
        f"expected {len(client_pool_df)}."
    )

if len(set(assigned_indices)) != len(assigned_indices):
    raise RuntimeError(
        "Duplicate rows were assigned to multiple clients."
    )


# ============================================================
# Save client files and distribution summary
# ============================================================

summary_rows = []


for client_number in range(1, NUM_CLIENTS + 1):
    indices = np.asarray(
        client_indices[client_number - 1],
        dtype=np.int64,
    )

    rng.shuffle(indices)

    client_df = client_pool_df.loc[
        indices
    ].reset_index(drop=True)

    output_path = CLIENT_PATH_TEMPLATE.format(
        client_id=client_number
    )

    client_df.to_csv(
        output_path,
        index=False,
    )

    print(
        f"Client {client_number} saved to {output_path} "
        f"({len(client_df):,} rows)"
    )

    counts = client_df["Label"].value_counts()

    for class_name in classes:
        summary_rows.append({
            "client": f"Client{client_number}",
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
    f"Client{client_number}"
    for client_number in range(1, NUM_CLIENTS + 1)
]

pivot["present_in"] = (
    pivot[client_columns] > 0
).sum(axis=1)

pivot["heterogeneity_score"] = (
    pivot[client_columns].std(axis=1)
    / (pivot[client_columns].mean(axis=1) + 1.0)
)

pivot = pivot.sort_values(
    "heterogeneity_score",
    ascending=False,
)

pivot.to_csv(SUMMARY_PATH)

print("\n" + "=" * 100)
print(
    f"EIGHT-CLASS DIRICHLET SPLIT "
    f"| alpha={ALPHA} | clients={NUM_CLIENTS}"
)
print("=" * 100)

print(
    pivot[
        client_columns
        + ["present_in", "heterogeneity_score"]
    ].to_string()
)

print(f"\nDistribution saved to {SUMMARY_PATH}")