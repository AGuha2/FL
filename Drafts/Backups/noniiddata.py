# create_noniid_splits_v2.py
import pandas as pd
import numpy as np

print("Loading CSVs...")
df = pd.concat([
    pd.read_csv(r'D:\CAPSTONE\Merged01.csv'),
    pd.read_csv(r'D:\CAPSTONE\Merged02.csv'),
    pd.read_csv(r'D:\CAPSTONE\Merged03.csv'),
    pd.read_csv(r'D:\CAPSTONE\Merged04.csv'),
    pd.read_csv(r'D:\CAPSTONE\Merged05.csv'),
]).dropna()
df['Label'] = df['Label'].str.strip().str.upper()
print(f"Total rows: {len(df):,}")

# Attack families
DDOS_CLASSES = [
    'DDOS-ACK_FRAGMENTATION','DDOS-HTTP_FLOOD','DDOS-ICMP_FLOOD',
    'DDOS-ICMP_FRAGMENTATION','DDOS-PSHACK_FLOOD','DDOS-RSTFINFLOOD',
    'DDOS-SLOWLORIS','DDOS-SYNONYMOUSIP_FLOOD','DDOS-SYN_FLOOD',
    'DDOS-TCP_FLOOD','DDOS-UDP_FLOOD','DDOS-UDP_FRAGMENTATION'
]
DOS_CLASSES = [
    'DOS-HTTP_FLOOD','DOS-SYN_FLOOD','DOS-TCP_FLOOD','DOS-UDP_FLOOD'
]
MIRAI_CLASSES = [
    'MIRAI-GREETH_FLOOD','MIRAI-GREIP_FLOOD','MIRAI-UDPPLAIN'
]
RECON_CLASSES = [
    'RECON-HOSTDISCOVERY','RECON-OSSCAN','RECON-PINGSWEEP','RECON-PORTSCAN'
]
RARE_CLASSES = [
    'XSS','SQLINJECTION','BACKDOOR_MALWARE','COMMANDINJECTION',
    'BROWSERHIJACKING','UPLOADING_ATTACK','VULNERABILITYSCAN',
    'DICTIONARYBRUTEFORCE','MITM-ARPSPOOFING','DNS_SPOOFING','BENIGN'
]

# 5 DISTINCT families — no overlap between clients
CLIENT_FAMILIES = {
    'Client1': DDOS_CLASSES,        # External gateway — flood attacks
    'Client2': DOS_CLASSES,         # Server-side — denial of service
    'Client3': MIRAI_CLASSES,       # IoT botnet devices
    'Client4': RECON_CLASSES,       # Internal scanner — reconnaissance
    'Client5': RARE_CLASSES,        # App layer — web attacks + rare classes
}

OUTPUT_PATHS = {
    f'Client{i}': rf'D:\CAPSTONE\NonIID_Client{i}.csv'
    for i in range(1, 6)
}

# Cap each client at 300k rows max so sizes are comparable
MAX_ROWS     = 300_000
DOMINANT_PCT = 0.70
SEED_BASE    = 42

def build_client_df(dominant_classes, all_df,
                    dominant_pct=0.70, max_rows=300_000, seed=42):
    dominant = all_df[all_df['Label'].isin(dominant_classes)]
    other    = all_df[~all_df['Label'].isin(dominant_classes)]

    # Work out how many dominant vs other rows to take
    n_dominant = int(max_rows * dominant_pct)
    n_other    = max_rows - n_dominant

    n_dominant = min(n_dominant, len(dominant))
    n_other    = min(n_other,    len(other))

    df_dom   = dominant.sample(n=n_dominant, random_state=seed)
    df_other = other.sample(n=n_other,    random_state=seed)

    return pd.concat([df_dom, df_other]) \
             .sample(frac=1, random_state=seed) \
             .reset_index(drop=True)

# Build all 5
print("\nBuilding non-IID splits...\n")
summary_rows = []

for i, (client_name, dominant_classes) in enumerate(CLIENT_FAMILIES.items()):
    seed      = SEED_BASE + i
    client_df = build_client_df(dominant_classes, df,
                                dominant_pct=DOMINANT_PCT,
                                max_rows=MAX_ROWS,
                                seed=seed)
    out_path = OUTPUT_PATHS[client_name]
    client_df.to_csv(out_path, index=False)
    print(f"✓ {client_name} saved → {out_path}  ({len(client_df):,} rows)")

    for label, count in client_df['Label'].value_counts().items():
        summary_rows.append({
            'client': client_name,
            'label':  label,
            'count':  count
        })

# Diagnostic
print("\n" + "="*75)
print("DIAGNOSTIC — class distribution per client")
print("="*75)

summary = pd.DataFrame(summary_rows)
pivot   = summary.pivot(
    index='label', columns='client', values='count'
).fillna(0).astype(int)
pivot['present_in'] = (pivot[list(CLIENT_FAMILIES.keys())] > 0).sum(axis=1)
pivot = pivot.sort_values('present_in')

pd.set_option('display.max_rows', 50)
pd.set_option('display.width', 130)
print(pivot.to_string())

print("\n" + "="*75)
print("DATASET SIZES + DOMINANT % ")
print("="*75)
for client_name, dominant_classes in CLIENT_FAMILIES.items():
    total    = pivot[client_name].sum()
    dominant = pivot.loc[
        pivot.index.isin(dominant_classes), client_name
    ].sum()
    pct = 100 * dominant / total if total > 0 else 0
    print(f"{client_name}: {total:>8,} rows | "
          f"dominant: {dominant:>7,} ({pct:.1f}%)")

print("\n" + "="*75)
print("TOP 3 CLASSES PER CLIENT")
print("="*75)
for client_name in CLIENT_FAMILIES:
    top3 = pivot[client_name].sort_values(ascending=False).head(3)
    print(f"\n{client_name}  ({CLIENT_FAMILIES[client_name][0]} family):")
    for label, count in top3.items():
        print(f"  {label:<35} {count:>8,}")

print("\nDone.")