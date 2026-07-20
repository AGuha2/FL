import pandas as pd
from Labels import ALL_LABELS

files = {
    'Client1': r'D:\CAPSTONE\Dir_Client1.csv',
    'Client2': r'D:\CAPSTONE\Dir_Client2.csv',
    'Client3': r'D:\CAPSTONE\Dir_Client3.csv',
    'Client4': r'D:\CAPSTONE\Dir_Client4.csv',
    'Client5': r'D:\CAPSTONE\Dir_Client5.csv'
}

coverage = {}
for name, path in files.items():
    df = pd.read_csv(path).dropna()
    counts = df['Label'].str.strip().str.upper().value_counts()
    coverage[name] = counts

summary = pd.DataFrame(coverage).fillna(0).astype(int)
summary['present_in'] = (summary > 0).sum(axis=1)
print(summary.sort_values('present_in'))