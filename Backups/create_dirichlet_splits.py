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

# Verify all labels 
unseen = set(df['Label']) - set(le.classes_)
if unseen:
    df = df[df['Label'].isin(le.classes_)].reset_index(drop=True)
    print(f"Rows after filtering: {len(df):,}")

# Verify feature count
feature_cols = df.drop(columns=['Label']).columns.tolist()
assert len(feature_cols) == NUM_FEATURES, \
    f"Expected {NUM_FEATURES} features, got {len(feature_cols)}"


#Fit ONE global scaler on the full dataset
scaler = RobustScaler()
scaler.fit(df[feature_cols])
scaler_path = r'D:\CAPSTONE\global_scaler.pkl'
joblib.dump(scaler, scaler_path)

#Parameters
NUM_CLIENTS = 5
ALPHA       = 0.5     
MAX_ROWS    = 300_000 
SEED        = 42
np.random.seed(SEED)

# Group row indices by class
classes = sorted(df['Label'].unique())
print(f"\nUnique classes in data: {len(classes)}")

class_indices = {
    c: df.index[df['Label'] == c].tolist()
    for c in classes
}

# Dirichlet partitioning
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

# Build and save client dataframes
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

# Diagnostic
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
