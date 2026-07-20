import pandas as pd
import numpy as np
import os


# Load and merge all datasets

files = [
    r'D:\CAPSTONE\Merged01.csv',
    r'D:\CAPSTONE\Merged02.csv',
    r'D:\CAPSTONE\Merged03.csv',
    r'D:\CAPSTONE\Merged04.csv',
    r'D:\CAPSTONE\Merged05.csv'
]

df = pd.concat([pd.read_csv(f) for f in files], ignore_index=True)

df['Label'] = df['Label'].astype(str).str.strip().str.upper()

print(f"Total dataset size: {len(df):,}")


# Dirichlet settings

num_clients = 5
alpha = 0.3   # lower = more non-IID
output_dir = r"D:\CAPSTONE\clients_dirichlet"

os.makedirs(output_dir, exist_ok=True)

client_indices = [[] for _ in range(num_clients)]

labels = df['Label'].unique()


# Dirichlet split per class

for label in labels:

    idx = df[df['Label'] == label].index.to_numpy().copy()
    np.random.shuffle(idx)

    proportions = np.random.dirichlet(
        np.repeat(alpha, num_clients)
    )

    proportions = (np.cumsum(proportions) * len(idx)).astype(int)[:-1]

    split_idx = np.split(idx, proportions)

    for client_id in range(num_clients):
        client_indices[client_id].extend(split_idx[client_id])

# Save clients

for i in range(num_clients):

    client_df = df.loc[client_indices[i]]

    save_path = os.path.join(output_dir, f'Client{i+1}.csv')

    client_df.to_csv(save_path, index=False)

    print(f"Client {i+1}: {len(client_df):,} samples → {save_path}")

print("\nDirichlet non-IID clients created.")