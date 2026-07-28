import pandas as pd
from sklearn.preprocessing import LabelEncoder

df1 = pd.read_csv(r'D:\CAPSTONE\Merged01.csv')
df2 = pd.read_csv(r'D:\CAPSTONE\Merged02.csv')
df3 = pd.read_csv(r'D:\CAPSTONE\Merged03.csv')

# Check each client's encoding independently
for i, df in enumerate([df1, df2, df3], 1):
    le = LabelEncoder()
    le.fit_transform(df['Label'])
    print(f"\nClient {i} label mapping:")
    for idx, label in enumerate(le.classes_):
        print(f"  y={idx} → {label}")