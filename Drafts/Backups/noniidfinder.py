import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.spatial.distance import jensenshannon

# ======================================
# Load all client datasets
# ======================================

client_files = {
    "Client1": r'D:\CAPSTONE\clients_dirichlet\Client1.csv',
    "Client2": r'D:\CAPSTONE\clients_dirichlet\Client2.csv',
    "Client3": r'D:\CAPSTONE\clients_dirichlet\Client3.csv',
    "Client4": r'D:\CAPSTONE\clients_dirichlet\Client4.csv',
    "Client5": r'D:\CAPSTONE\clients_dirichlet\Client5.csv'
}

clients = {}

for name, path in client_files.items():
    clients[name] = pd.read_csv(path)

print("Datasets loaded successfully.")

# ======================================
# Get label distributions
# ======================================

label_counts = {}

for name, df in clients.items():
    label_counts[name] = df['Label'].value_counts()

# Collect all unique labels
all_labels = sorted(
    set().union(
        *[counts.index for counts in label_counts.values()]
    )
)

# ======================================
# Convert counts → probability distributions
# ======================================

probabilities = {}

for name, counts in label_counts.items():

    counts = counts.reindex(
        all_labels,
        fill_value=0
    )

    probabilities[name] = counts / counts.sum()

# ======================================
# Calculate JS distance matrix
# ======================================

client_names = list(probabilities.keys())

js_matrix = pd.DataFrame(
    index=client_names,
    columns=client_names
)

epsilon = 1e-10

for c1 in client_names:
    for c2 in client_names:

        p1 = probabilities[c1] + epsilon
        p2 = probabilities[c2] + epsilon

        js = jensenshannon(p1, p2)

        js_matrix.loc[c1, c2] = round(js,4)

# ======================================
# Print results
# ======================================

print("\n")
print("="*60)
print("Jensen-Shannon Distance Matrix")
print("="*60)

print(js_matrix)

# ======================================
# Compute average JS score
# ======================================

values=[]

for i in range(len(client_names)):
    for j in range(i+1,len(client_names)):
        values.append(
            float(js_matrix.iloc[i,j])
        )

avg_js=np.mean(values)

print("\nAverage JS Distance:",round(avg_js,4))

# ======================================
# Interpret result
# ======================================

if avg_js < 0.05:
    interpretation="Nearly IID"

elif avg_js <0.15:
    interpretation="Mildly Non-IID"

elif avg_js <0.30:
    interpretation="Moderately Non-IID"

else:
    interpretation="Strongly Non-IID"

print("\nDistribution Type:", interpretation)

# ======================================
# Save matrix
# ======================================

save_path = r'D:\CAPSTONE\JS_Distance_Matrix.csv'

js_matrix.to_csv(save_path)

print("\nSaved matrix:")
print(save_path)

# ======================================
# Visualize heatmap
# ======================================

plt.figure(figsize=(8,6))

matrix = js_matrix.astype(float)

plt.imshow(matrix)

plt.colorbar(
    label='JS Distance'
)

plt.xticks(
    range(len(client_names)),
    client_names
)

plt.yticks(
    range(len(client_names)),
    client_names
)

plt.title(
    "Non-IID Analysis Across 5 Clients"
)

for i in range(len(client_names)):
    for j in range(len(client_names)):
        plt.text(
            j,
            i,
            matrix.iloc[i,j],
            ha='center'
        )

plt.tight_layout()

plot_path=r'D:\CAPSTONE\JS_Heatmap.png'

plt.savefig(
    plot_path,
    dpi=300
)

plt.show()

print("\nSaved heatmap:")
print(plot_path)