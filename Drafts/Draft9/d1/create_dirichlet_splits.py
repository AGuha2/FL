import pandas as pd
import numpy as np
import joblib
from sklearn.preprocessing import RobustScaler
from Labels import le, NUM_CLASSES, NUM_FEATURES

# ── Load all source data ───────────────────────────────────────────────────────
print("Loading CSVs...")
df = pd.concat([
    pd.read_csv(r'D:\CAPSTONE\Merged01.csv'),
    pd.read_csv(r'D:\CAPSTONE\Merged02.csv'),
    pd.read_csv(r'D:\CAPSTONE\Merged03.csv'),
    pd.read_csv(r'D:\CAPSTONE\Merged04.csv'),
    pd.read_csv(r'D:\CAPSTONE\Merged05.csv'),
]).dropna()

df.replace([np.inf, -np.inf], np.nan, inplace=True)
df = df.dropna()
df['Label'] = df['Label'].str.strip().str.upper()
df = df.reset_index(drop=True)
print(f"Total rows loaded: {len(df):,}")

# ── Verify all labels are in shared taxonomy ───────────────────────────────────
unseen = set(df['Label']) - set(le.classes_)
if unseen:
    print(f"WARNING: removing {len(unseen)} unseen labels: {unseen}")
    df = df[df['Label'].isin(le.classes_)].reset_index(drop=True)
    print(f"Rows after filtering: {len(df):,}")

# ── Verify feature count ───────────────────────────────────────────────────────
feature_cols = df.drop(columns=['Label']).columns.tolist()
assert len(feature_cols) == NUM_FEATURES, \
    f"Expected {NUM_FEATURES} features, got {len(feature_cols)}"
print(f"Features: {len(feature_cols)} ✓")
print(f"Classes:  {NUM_CLASSES} ✓")

# ── Fit ONE global scaler on the full dataset, before partitioning ────────────
# This is critical: if each client fits its own RobustScaler locally (as before),
# the same raw feature value maps to a different scaled value on every client,
# because each client sees a different (Dirichlet-skewed) slice of the data.
# When the server averages weights trained on different input spaces, the
# aggregated model is trained on data that's inconsistent client-to-client.
# Fitting once here and reusing everywhere fixes that.
print("\nFitting global RobustScaler on full dataset...")
scaler = RobustScaler()
scaler.fit(df[feature_cols])
scaler_path = r'D:\CAPSTONE\global_scaler.pkl'
joblib.dump(scaler, scaler_path)
print(f"✓ Global scaler saved → {scaler_path}")

# ── Parameters ────────────────────────────────────────────────────────────────
NUM_CLIENTS = 5
ALPHA       = 0.5     # change to 0.1 for stronger non-IID
MAX_ROWS    = 300_000 # cap per client so training time is manageable
SEED        = 42
np.random.seed(SEED)

# ── Group row indices by class ─────────────────────────────────────────────────
classes = sorted(df['Label'].unique())
print(f"\nUnique classes in data: {len(classes)}")

class_indices = {
    c: df.index[df['Label'] == c].tolist()
    for c in classes
}

# ── Dirichlet partitioning ─────────────────────────────────────────────────────
client_indices = [[] for _ in range(NUM_CLIENTS)]

for cls in classes:
    idx = class_indices[cls].copy()
    np.random.shuffle(idx)

    proportions = np.random.dirichlet(alpha=[ALPHA] * NUM_CLIENTS)

    splits      = (proportions * len(idx)).astype(int)
    splits[-1]  = len(idx) - splits[:-1].sum()  # fix rounding

    start = 0
    for k, n in enumerate(splits):
        client_indices[k].extend(idx[start:start + n])
        start += n

# ── Build and save client dataframes ──────────────────────────────────────────
print(f"\nBuilding {NUM_CLIENTS} Dirichlet(α={ALPHA}) client splits...\n")

summary_rows = []

for k in range(NUM_CLIENTS):
    idx = client_indices[k]
    np.random.shuffle(idx)

    client_df = df.loc[idx].reset_index(drop=True)

    if len(client_df) > MAX_ROWS:
        client_df = client_df.sample(
            n=MAX_ROWS, random_state=SEED + k
        ).reset_index(drop=True)

    client_unseen = set(client_df['Label']) - set(le.classes_)
    if client_unseen:
        print(f"  WARNING Client {k+1}: removing unseen labels {client_unseen}")
        client_df = client_df[
            client_df['Label'].isin(le.classes_)
        ].reset_index(drop=True)

    out_path = rf'D:\CAPSTONE\Dir_Client{k+1}.csv'
    client_df.to_csv(out_path, index=False)
    print(f"✓ Client {k+1} saved → {out_path}  ({len(client_df):,} rows)")

    for label, count in client_df['Label'].value_counts().items():
        summary_rows.append({
            'client': f'Client{k+1}',
            'label':  label,
            'count':  count
        })

# ── Diagnostic ─────────────────────────────────────────────────────────────────
print("\n" + "="*80)
print(f"DIAGNOSTIC — Dirichlet α={ALPHA} | {NUM_CLIENTS} clients")
print("="*80)

summary = pd.DataFrame(summary_rows)
pivot   = summary.pivot(
    index='label', columns='client', values='count'
).fillna(0).astype(int)

client_cols = [f'Client{k+1}' for k in range(NUM_CLIENTS)]

pivot['het_score'] = pivot[client_cols].std(axis=1) / \
                     (pivot[client_cols].mean(axis=1) + 1)

pivot = pivot.sort_values('het_score', ascending=False)

pd.set_option('display.max_rows', 50)
pd.set_option('display.width', 130)
print(pivot[client_cols + ['het_score']].to_string())

print("\n" + "="*80)
print("DATASET SIZES")
print("="*80)
for k in range(NUM_CLIENTS):
    total    = pivot[f'Client{k+1}'].sum()
    n_labels = (pivot[f'Client{k+1}'] > 0).sum()
    print(f"Client {k+1}: {total:>8,} rows | {n_labels} distinct classes")

print("\n" + "="*80)
print("TOP 5 CLASSES PER CLIENT — confirms non-IID skew")
print("="*80)
for k in range(NUM_CLIENTS):
    col  = f'Client{k+1}'
    top5 = pivot[col].sort_values(ascending=False).head(5)
    total = pivot[col].sum()
    print(f"\nClient {k+1}:")
    for label, count in top5.items():
        pct = 100 * count / total if total > 0 else 0
        print(f"  {label:<38} {count:>8,}  ({pct:.1f}%)")

print("\n" + "="*80)
print("RARE CLASS CHECK — classes with < 10 samples per client")
print("="*80)
for k in range(NUM_CLIENTS):
    col  = f'Client{k+1}'
    rare = pivot[pivot[col] < 10][col]
    rare = rare[rare > 0]
    if len(rare) > 0:
        print(f"Client {k+1} rare classes (<10 samples): "
              f"{list(rare.index)}")
    else:
        print(f"Client {k+1}: no rare classes")

print(f"\n{'='*80}")
print(f"Done. CSV files saved to D:\\CAPSTONE\\Dir_Client1-5.csv")
print(f"Global scaler saved to {scaler_path}")
print(f"Every client script must load THIS scaler with joblib and call .transform() only —")
print(f"never fit a new scaler locally.")