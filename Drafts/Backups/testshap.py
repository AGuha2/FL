# test_shap.py — fix: use larger sample and no stratify
import numpy as np
import pandas as pd
import shap
import time
from tensorflow import keras
from sklearn.preprocessing import LabelEncoder, RobustScaler
from sklearn.model_selection import train_test_split

ALL_LABELS = [
    "BACKDOOR_MALWARE", "BENIGN", "BROWSERHIJACKING",
    "COMMANDINJECTION", "DDOS-ACK_FRAGMENTATION",
    "DDOS-HTTP_FLOOD", "DDOS-ICMP_FLOOD",
    "DDOS-ICMP_FRAGMENTATION", "DDOS-PSHACK_FLOOD",
    "DDOS-RSTFINFLOOD", "DDOS-SLOWLORIS",
    "DDOS-SYNONYMOUSIP_FLOOD", "DDOS-SYN_FLOOD",
    "DDOS-TCP_FLOOD", "DDOS-UDP_FLOOD",
    "DDOS-UDP_FRAGMENTATION", "DICTIONARYBRUTEFORCE",
    "DNS_SPOOFING", "DOS-HTTP_FLOOD", "DOS-SYN_FLOOD",
    "DOS-TCP_FLOOD", "DOS-UDP_FLOOD", "MIRAI-GREETH_FLOOD",
    "MIRAI-GREIP_FLOOD", "MIRAI-UDPPLAIN",
    "MITM-ARPSPOOFING", "RECON-HOSTDISCOVERY",
    "RECON-OSSCAN", "RECON-PINGSWEEP", "RECON-PORTSCAN",
    "SQLINJECTION", "UPLOADING_ATTACK",
    "VULNERABILITYSCAN", "XSS",
]
le = LabelEncoder()
le.fit(ALL_LABELS)


# Load full CSV but sample 20000 rows
df = pd.read_csv(r'D:\CAPSTONE\Merged01.csv')
df.replace([np.inf, -np.inf], np.nan, inplace=True)
df = df.dropna()

# Sample per class to ensure all 34 classes represented
df = df.groupby('Label').apply(
    lambda x: x.sample(min(len(x), 600), random_state=42)
).reset_index(drop=True)

print(f"Sampled dataset size: {len(df)}")
print(f"Classes present: {df['Label'].nunique()}")

X = df.drop(columns=['Label']).values
y = le.transform(df['Label'])

NUM_CLASSES  = 34
NUM_FEATURES = X.shape[1]

scaler = RobustScaler()
X = scaler.fit_transform(X)

# No stratify — sample is already balanced per class
x_train, x_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

print(f"Train: {len(x_train)} | Test: {len(x_test)}")

# Build and train briefly
model = keras.Sequential([
    keras.layers.Input(shape=(NUM_FEATURES,)),
    keras.layers.Dense(128, activation='relu'),
    keras.layers.Dense(NUM_CLASSES, activation='softmax')
])
model.compile(
    optimizer='adam',
    loss='sparse_categorical_crossentropy',
    metrics=['accuracy']
)
print("Training briefly...")
model.fit(x_train, y_train, epochs=2, verbose=0)
print("Training done.")

# Test SHAP
print("\nComputing SHAP (may take 1-3 minutes)...")
start = time.time()

background = x_train[np.random.choice(
    len(x_train), 30, replace=False
)]
explain = x_train[np.random.choice(
    len(x_train), 30, replace=False
)]

explainer   = shap.KernelExplainer(model.predict, background)
shap_values = explainer.shap_values(explain, nsamples=100)

//shap.summary_plot(shap_values, x_test)

phi_k = np.mean(
    [np.abs(sv).mean(axis=0) for sv in shap_values],
    axis=0
)

elapsed = time.time() - start
print(f"\nSHAP done in {elapsed:.1f}s")
print(f"Vector shape  : {phi_k.shape}")
print(f"Any NaN?      : {np.any(np.isnan(phi_k))}")
print(f"All zero?     : {np.all(phi_k == 0)}")
print(f"Top 5 feature indices : {np.argsort(phi_k)[::-1][:5]}")
print(f"Top 5 values          : {np.round(phi_k[np.argsort(phi_k)[::-1][:5]], 6)}")